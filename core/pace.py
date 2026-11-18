"""语速均衡：让全片每个发声窗口的「字数 / 有效发声时长」尽量一致，
同时把「段与段之间的停顿」也拉成自然的小停顿，而不是忽长忽短。

旧版的两个结构性毛病（已修）：
  1. core/vad.py 整句 ASR 按标点切 units，units 与窗口数对不上就按窗口时长硬切字数，
     于是「短窗口塞一堆字」「长窗口只放几个字」并存。
  2. core/align.py 旧版用相位声码器把每段 TTS 强制铺满窗口，字少被拖长、字多被压快。

本模块（v3，更聪明的节奏控制）：
  * 每段「可用时长」slot = 自身窗口 + 可借的后续静音（最长到下一段起点 - guard）。
  * 全片统一 unit（秒/字），使「总发声 + 总目标停顿」恰好铺满全部 slot 之和，
    因此不存在「字少被迫拖长」或「窗口空着好几秒不说话」。
  * 组内按标点吸附切点；再用 _balance_pauses 在组内搬移字数，
    让每段停顿都贴近目标（句内 target_mid 小停顿，句尾 target_bnd 略大），
    既不会某段空好几秒，也不会下一段贴脸零间隔。
  * TTS 参数域只按 voice_end 调时长，波形不再经过相位声码器（见 align.py）。

硬约束不变：段起点不动、段与段不重叠、整轨总长 = 视频长。
"""
import re
import numpy as np

_PUNCT = r"[，,、；;。！!？?：:\s]+"

# 常见不可拆的双/三字词：切点落进这些词中间时，左右挪一字错开
_NO_SPLIT = (
    "有可能 电动车 自行车 锂电池 充电棚 室内 室外 一旦 引燃 爆燃 扑救 难度 "
    "烟气 有毒 疏散 逃生 蔓延 隐患 安全 发生 意外 自燃 快速 大量 释放 "
    "及时 很难 过程 大家 远离 守护 全家 平安 朋友 省事 推进 自家 屋内 "
    "密闭 发展 极快 迅速 风险 专用 日常 生活 做法 存在 极大 同时 人员 "
    "更有可能 就会 火灾 火情 充电过程 过程中 安全隐患 电瓶车 "
    "直接把 推进 自家屋内充电"
).split()


def _han(s):
    return re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", s or "")


def _boundaries(text):
    """返回 (纯文本, 标点后的字符位置列表)。位置 = 第几个字之后。"""
    clean, cuts = "", []
    pos = 0
    for m in re.finditer(_PUNCT, text or ""):
        clean += _han((text or "")[pos:m.start()])
        pos = m.end()
        if clean:
            cuts.append(len(clean))
    clean += _han((text or "")[pos:])
    return clean, [c for c in cuts if 0 < c < len(clean)]


def _avoid_word_split(clean, cut, lo, hi):
    """切点避开词内部：若 cut 落在某个词中间，挪到该词起点或终点。"""
    best = None
    for w in _NO_SPLIT:
        start = 0
        while True:
            s = clean.find(w, start)
            if s < 0:
                break
            e = s + len(w)
            if s < cut < e:
                for cand in (s, e):
                    d = abs(cand - cut)
                    if lo <= cand <= hi and d <= 2 and (best is None or d < best[0]):
                        best = (d, cand)
            start = s + 1
    return best[1] if best else cut


def _assign(clean, counts, punc_cuts, n):
    """按 counts 切 n 段，切点优先吸附标点、再避开词内部。"""
    cuts, acc = [], 0
    for k in range(n - 1):
        acc += counts[k]
        exact = acc
        near = [c for c in punc_cuts if abs(c - exact) <= 2 and
                (not cuts or c > cuts[-1]) and len(clean) - c >= n - k - 1]
        c = near[0] if near else exact
        lo = (cuts[-1] + 1) if cuts else 1
        hi = len(clean) - (n - k - 1)
        c = max(lo, min(c, hi))
        c = _avoid_word_split(clean, c, lo, hi)
        cuts.append(c)
    b = [0] + cuts + [len(clean)]
    return [clean[b[k]:b[k + 1]] for k in range(n)]


def _balance_pauses(counts, idxs, slot, tgt, unit, guard=0.05, iters=600):
    """把同组内的字数在段落间搬移，使每段「实发时长 + 目标停顿」≈ 该段可用时长，
    即让每段停顿都贴近目标值（句内小停顿、句尾略大），从而消除
    「某段窗口空着好几秒不说话」与「下一段零间隔被挤成一团」两种观感。

    单调性保证：每次只接受能降低「(实际停顿 - 目标停顿)² 之和」的搬移，
    因此必然收敛，不会在两段之间来回震荡。
    """
    counts = [int(c) for c in counts]
    n = len(idxs)

    def pause(k):
        return slot[idxs[k]] - counts[k] * unit

    def se():
        return float(sum((pause(k) - tgt[idxs[k]]) ** 2 for k in range(n)))

    for _ in range(iters):
        cur = se()
        best, bk, bl = 0.0, -1, -1
        for hi in range(n):                      # 接收方：停顿过大（死气）
            if pause(hi) <= tgt[idxs[hi]] + 1e-3:
                continue
            for lo in range(n):                  # 出让方：停顿过小且有字可让
                if lo == hi or counts[lo] <= 1:
                    continue
                if pause(lo) >= tgt[idxs[lo]] - 1e-3:
                    continue
                counts[lo] -= 1
                counts[hi] += 1
                d = cur - se()
                counts[hi] -= 1
                counts[lo] += 1
                if d > best + 1e-6:
                    best, bk, bl = d, hi, lo
        if bk < 0 or best <= 1e-6:
            break
        counts[bl] -= 1
        counts[bk] += 1
    return counts


def balance(segs, srt, borrow=0.90, guard=0.05, log=print,
            target_mid=0.18, target_bnd=0.55):
    """全片统一字速 + 逐段停顿拉齐，返回 (新 segments, 统计)。

    borrow     : 单段最多向后借用多少秒静音（不越过下一段 start - guard）
    target_mid : 句内每段期望停顿（秒），避免零间隔挤成一团
    target_bnd : 句尾（组内最后一段）期望停顿（秒），自然句间留白
    """
    segs = [dict(s) for s in segs]
    N = len(segs)
    if N == 0:
        return segs, []
    total = float(max((s["end"] for s in segs), default=0.0))

    # ---- 组归并（旧时间轴无 group 时按 srt 区间补） ----
    if all(int(s.get("group", 0)) == int(segs[0].get("group", 0)) for s in segs) and (srt or []):
        for i, s in enumerate(segs):
            mid = (s["start"] + s["end"]) / 2
            gi = 0
            for k, (a, b, _) in enumerate(srt):
                if a - 0.3 <= mid <= b + 0.3:
                    gi = k
                    break
            s["group"] = gi
    groups = {}
    for s in segs:
        groups.setdefault(int(s.get("group", 0)), []).append(s)

    # ---- 每组完整台词（优先用 srt 带标点；与用户手改不一致时以手改拼接为准） ----
    joined = {gi: "".join(s["text"] for s in g) for gi, g in groups.items()}
    full = {}
    for gi in groups:
        cands = [t for k, (a, b, t) in enumerate(srt or [])
                 if k == gi or (abs(groups[gi][0]["start"] - a) < 1.8
                                and abs(groups[gi][-1]["end"] - b) < 1.8)]
        hit = next((t for t in cands if _han(t) == _han(joined[gi])), None)
        full[gi] = hit or "，".join(s["text"] for s in groups[gi])

    # ---- 每段可用时长 slot ----
    starts = [s["start"] for s in segs]
    nxt = [segs[i + 1]["start"] if i + 1 < N else total for i in range(N)]
    slot = []
    for i in range(N):
        room = max(0.0, nxt[i] - segs[i]["end"] - guard)
        slot.append(max(0.05, (segs[i]["end"] - segs[i]["start"]) + min(borrow, room)))

    # 句尾（组内最后一段）允许更大停顿，其余句内小停顿
    last_in_group = set(id(g[-1]) for g in groups.values() if g)
    tgt = [(target_bnd if id(s) in last_in_group else target_mid) for s in segs]

    total_zi = sum(len(_han(s["text"])) for s in segs)
    sum_slot = float(sum(slot))
    sum_tgt = float(sum(tgt))
    if total_zi == 0:
        for s in segs:
            s["voice_end"] = s["end"]
        return segs, []

    # 统一 unit：让「总发声 + 总目标停顿」恰好铺满所有 slot 之和，
    # 这样全片既不会字少拖长、也不会某段窗口空着不说话。
    unit = max(sum_slot - sum_tgt, total_zi * 0.1) / total_zi
    cps = 1.0 / unit
    cps = min(max(cps, 3.0), 7.5)                 # 中文舒适语速带，避免过慢/过快
    unit = 1.0 / cps

    # ---- 按组：切点吸附标点 → 字数按 slot 比例初分 → 停顿拉齐 → 回写 ----
    for gi, g in groups.items():
        changed = any(s.get("locked") for s in g)
        if changed:
            # 用户在时间轴 JSON 里给本组某段标了 "locked": true：
            # 尊重手改，保留每段 text 与 voice_end 原样，不重切、不搬字，
            # 后续全局步骤也不会覆写 locked 段的 voice_end。
            log(f"[均衡] 第 {gi + 1} 句：含 locked 标记，保留你手改的文字与 voice_end")
            continue
        clean, punc = _boundaries(full[gi])
        idxs = [segs.index(s) for s in g]
        L = len(clean)
        if L == 0:
            for i in idxs:
                segs[i]["text"] = ""
            continue
        caps = [slot[i] for i in idxs]
        D = float(sum(caps))
        want = [L * c / D for c in caps]
        counts = [max(1, int(round(w))) for w in want]
        while sum(counts) > L and max(counts) > 1:
            k = int(np.argmax([counts[j] - want[j] for j in range(len(counts))]))
            counts[k] -= 1
        while sum(counts) < L:
            k = int(np.argmax([want[j] - counts[j] for j in range(len(counts))]))
            counts[k] += 1
        counts = _balance_pauses(counts, idxs, slot, tgt, unit, guard=guard)
        parts = _assign(clean, counts, punc, len(idxs))
        for i, p in zip(idxs, parts):
            segs[i]["text"] = p

    # ---- 实发时长与停顿统计 + voice_end ----
    stats = []
    for i, s in enumerate(segs):
        if s.get("locked") and s.get("voice_end"):
            s["voice_end"] = round(float(s["voice_end"]), 4)
            continue
        zi = len(_han(s["text"]))
        need = zi * unit
        ve = min(starts[i] + need, starts[i] + slot[i])
        ve = max(ve, starts[i] + 0.05)
        s["voice_end"] = round(ve, 4)
        gap = round(nxt[i] - ve, 3)
        stats.append({"id": s["id"], "zi": zi,
                      "window": round(s["end"] - s["start"], 3),
                      "used": round(ve - s["start"], 3),
                      "cps": round(zi / max(ve - s["start"], 1e-6), 2),
                      "gap": gap,
                      "text": s["text"]})

    gaps = [r["gap"] for r in stats]
    v = [r["cps"] for r in stats if r["zi"] > 0]
    log(f"[均衡] {len(stats)} 段  目标 {cps:.2f} 字/秒  实际 {min(v):.2f}~{max(v):.2f}"
        f"  句内停顿≈{target_mid}s / 句尾≈{target_bnd}s  最大停顿 {max(gaps):.2f}s")
    return segs, stats
