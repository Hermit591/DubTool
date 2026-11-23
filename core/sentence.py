"""句子级波形定长 + 弹性停顿（替代逐段变速）。

设计要点（v4，修复「碎片 + 重叠漂移」）：
  * 每句一次整段合成，整句波形作为一个**连续块**铺轨，句内绝不切分。
    v3 曾把整句波形按字数比例切成 N 段塞回原分段结构，切点落在字词中间
    （吞字=碎片），且 voice_end 用「句首+连续偏移」算、铺轨起点用各段原始
    start（带静音间隔），两者错位导致相邻段波形重叠叠加（漂移+金属音）。
    v4 彻底取消按段切分：一句 = 一个连续块，从句子起点铺 native 秒。
  * 句子起点锚定原声窗口起点 = 画面切换点，消除音画轻微错位。
  * 发声时长由合成波形的真实长度决定；当波形长过「到下一句起点」的
    可用空间时，只整句统一微调一次（限幅 max_adjust），句内依旧同速。
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
         max_adjust=0.15, min_gap=0.08, headroom=0.99):
    """产出「句子连续块」列表，不再按段切波形。

    返回 (blocks, retune)：
      blocks[k] = {"start": 句子起点秒, "wave": np.float32, "sr": int,
                   "native": 真实发声秒, "text": 整句文本, "group": 组号,
                   "id": 块 id, "adjust_pct": 整句微调百分比}
      每个块在 align.build_track 里作为一个连续区间铺轨，块内不切、不补零。
      retune = 句子级微调记录（供报告/日志）。
    """
    P = len(order)
    blocks, retune = [], []

    # 统一去引擎前后静音：停顿由时间轴负责，不能让 TTS 自己加。
    trims = []
    for w, sr in sent_audios:
        trims.append((align.trim_silence(np.asarray(w, dtype=np.float32), sr), sr))

    base_pct = float(getattr(engine, "base_pct", 0.0) or 0.0)

    for pos, g in enumerate(order):
        idxs = groups[g]
        st = float(segs[idxs[0]]["start"])
        nxt = (float(segs[groups[order[pos + 1]][0]]["start"])
               if pos + 1 < P else float(total))
        # 可用时长：本句起点到下一句起点之间，留 min_gap 最小间隔。
        avail = max(0.3, nxt - st - min_gap)
        w, sr = trims[pos]
        native = len(w) / sr if sr else 0.0

        # 整句统一微调一次（只加速方向；句内绝不变速）。
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

        # 二次保险：微调后仍超出可用空间的极端情况，尾部裁剪到 avail，
        # 绝不让本句波形越过下一句起点造成重叠（宁可吞掉一点点尾音）。
        cap_n = int(avail * sr)
        if cap_n > 0 and len(w) > cap_n:
            w = w[:cap_n]
            native = len(w) / sr

        blocks.append({
            "id": f"g{g + 1:02d}", "group": g, "start": round(st, 4),
            "wave": w, "sr": sr, "native": round(native, 4),
            "text": sent_texts[pos], "adjust_pct": round(adj * 100.0, 1),
        })
        retune.append({"group": g, "native": round(native, 3), "avail": round(avail, 3),
                       "adjust": round(adj * 100.0, 1), "start": round(st, 3)})

    if retune:
        ad = [r["adjust"] for r in retune]
        log(f"     句子级定长：{len(retune)} 句，整块连续铺轨，句内不切分；"
            f"句级微调 {min(ad):+.0f}~{max(ad):+.0f}%（上限 {max_adjust*100:.0f}%）")
    return blocks, retune
