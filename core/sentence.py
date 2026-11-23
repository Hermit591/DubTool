"""句子级波形定长 + 弹性停顿（替代逐段变速）。

设计要点：
  * 每句一次整段合成，句内保持 TTS 自然语速 —— 不再逐段拧 rate。
    逐段变速正是「时快时慢」的根源（实测同句内 -13.8% ~ +37.3% 横跳）。
  * 句子起点锚定原声窗口起点 = 画面切换点，消除音画轻微错位。
  * 发声时长由合成波形的真实长度决定；当波形长过「到下一句起点」的
    可用空间时，整句统一微调一次（限幅 max_adjust），句内依旧同速。
  * 句尾多出/少下的时间由原窗口之后的静音间隔弹性吸收。
"""
import numpy as np
from . import align


def group_indices(segs):
    """按 group 号聚合段下标，返回 (groups, 有序 group 号列表)。"""
    groups, order = {}, []
    for i, s in enumerate(segs):
        g = int(s.get("group", 0))
        if g not in groups:
            groups[g] = []
            order.append(g)
        groups[g].append(i)
    return groups, order


def _han_len(t):
    return max(1, len([c for c in (t or "") if "\u4e00" <= c <= "\u9fff"]))


def plan(segs, order, groups, sent_audios, sent_texts, engine, total, log=print,
         max_adjust=0.15, min_gap=0.06, headroom=0.99):
    """把整句波形按字数比例切给各段，重算每段 voice_end。

    返回 (audios, retune)：
      audios[i] = (wave, sr) 与 segs 一一对应；
      retune   = 句子级微调记录（供报告/日志）。
    """
    N, P = len(segs), len(order)
    audios = [None] * N
    retune = []

    # 先统一去引擎前后静音：停顿由时间轴负责，不能让 TTS 自己加
    trims = []
    for w, sr in sent_audios:
        trims.append((align.trim_silence(np.asarray(w, dtype=np.float32), sr), sr))

    base_pct = float(getattr(engine, "base_pct", 0.0) or 0.0)

    for pos, g in enumerate(order):
        idxs = groups[g]
        st = float(segs[idxs[0]]["start"])
        nxt = (float(segs[groups[order[pos + 1]][0]]["start"])
               if pos + 1 < P else float(total))
        # 可用时长：本句起点到下一句起点之间，留 min_gap 最小间隔
        avail = max(0.3, nxt - st - min_gap)
        w, sr = trims[pos]
        native = len(w) / sr if sr else 0.0

        # 整句统一微调一次（只加速方向；句内绝不变速）
        adj = 0.0
        if native > avail * headroom and native > 0.5:
            adj = float(np.clip(native / (avail * headroom) - 1.0, 0.0, max_adjust))
        if adj >= 0.01:
            pct = base_pct + adj * 100.0
            try:
                got = engine.resynth_many([(pos, pct)], sent_texts, tag="sent")
                if pos in got:
                    w2, sr2 = got[pos]
                    trims[pos] = (align.trim_silence(np.asarray(w2, dtype=np.float32), sr2), sr2)
            except Exception as e:
                log(f"     句子 g{g} 级微调跳过（{type(e).__name__}），保留原速")
        w, sr = trims[pos]
        native = len(w) / sr if sr else 0.0

        # 按字数比例把整句波形切给各段；切点只作数组边界，播放首尾相接不断音
        weights = np.array([_han_len(sent_texts[i] if i < len(sent_texts) else segs[i]["text"])
                            for i in idxs], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(idxs), dtype=float)
        weights /= weights.sum()
        cursor = 0
        for k, i in enumerate(idxs):
            n = int(round(len(w) * weights[k]))
            if k == len(idxs) - 1:
                n = len(w) - cursor
            n = max(0, n)
            audios[i] = (w[cursor:cursor + n], sr)
            segs[i]["voice_end"] = round(st + (cursor + n) / sr, 4)
            cursor += n

        retune.append({"group": g, "native": round(native, 3), "avail": round(avail, 3),
                       "adjust": round(adj * 100.0, 1), "start": round(st, 3)})

    if retune:
        ad = [r["adjust"] for r in retune]
        log(f"     句子级定长：{len(retune)} 句，句内同速，"
            f"句级微调 {min(ad):+.0f}~{max(ad):+.0f}%（上限 {max_adjust*100:.0f}%）")
    return audios, retune
