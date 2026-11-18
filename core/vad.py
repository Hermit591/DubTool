"""第 2 步：只看波形，不听内容——用能量/静音切出每个发声窗口。

这是「一秒不差」的核心：模型完全不接触原音频，窗口起止由波形决定，
文字是后面 ASR 单独识别、再分配到这些窗口里的。
"""
import re
import numpy as np


def _true_runs(mask: np.ndarray) -> list:
    """把布尔序列切成 [(start_idx, end_idx_exclusive), ...] 的 True 区间。"""
    m = np.asarray(mask, dtype=bool)
    d = np.diff(np.concatenate(([False], m, [False])).astype(int))
    starts = np.where(d == 1)[0]
    ends = np.where(d == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def detect_chunks(y: np.ndarray, sr: int, noise_db: float = -38,
                  adaptive_ratio: float = 0.04, min_silence: float = 0.20) -> list:
    """返回发声窗口 [(start_sec, end_sec), ...]。

    20ms 帧 RMS；阈值取绝对 -38dBFS 与「95 分位能量 × ratio」的较大者，
    这样响度不同的素材都能切对。短于 min_silence 的空隙并入发声段。
    """
    y = np.asarray(y, dtype=np.float32)
    hop = int(sr * 0.02)
    n = len(y) // hop
    if n < 10:
        return [(0.0, len(y) / sr)]
    rms = np.sqrt(np.mean(y[:n * hop].reshape(n, hop) ** 2, axis=1))
    peak = float(np.percentile(rms, 95))
    thr = max(10 ** (noise_db / 20), peak * adaptive_ratio)

    raw = [(a * hop / sr, b * hop / sr) for a, b in _true_runs(rms > thr)]
    merged = []
    for a, b in raw:
        if merged and a - merged[-1][1] < min_silence:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    return [(round(a, 4), round(min(b, len(y) / sr), 4)) for a, b in merged if b - a > 0.08]


def group_sentences(chunks: list, speaker_gap: float = 0.60) -> list:
    """按较长停顿把发声窗口归成句子组：每组送一次 ASR，组内再按字数分配回窗口。"""
    if not chunks:
        return []
    groups, cur = [], [0]
    for i in range(1, len(chunks)):
        gap = chunks[i][0] - chunks[i - 1][1]
        if gap >= speaker_gap:
            groups.append(cur)
            cur = [i]
        else:
            cur.append(i)
    groups.append(cur)
    return groups


_PUNCT = r"[，,、；;。！!？?：:\s]+"


def assign_text(text: str, n: int, durations: list = None) -> list:
    """把一句识别结果拆成 n 段，尽量在标点处切，否则按窗口时长比例切。"""
    text = (text or "").strip()
    if n <= 1:
        return [re.sub(_PUNCT, "", text)]
    units = [u for u in re.split(_PUNCT, text) if u]
    if len(units) == n:
        return units
    if len(units) > n:
        units = list(units)
        while len(units) > n:
            pairs = [len(units[i]) + len(units[i + 1]) for i in range(len(units) - 1)]
            k = int(np.argmin(pairs))
            units[k] = units[k] + units[k + 1]
            del units[k + 1]
        return units

    clean = re.sub(r"[^\u4e00-\u9fa5A-Za-z0-9]", "", text)
    L = len(clean)
    if L == 0:
        return [""] * n
    w = list(durations) if durations and len(durations) == n else [1.0] * n
    total = float(sum(w))
    cuts, acc = [], 0.0
    for i in range(n - 1):
        acc += w[i]
        cuts.append(int(round(L * acc / total)))
    # 保证严格递增、不越界，每段至少 1 个字
    fixed = []
    for i, c in enumerate(cuts):
        lo = (fixed[-1] + 1) if fixed else 1
        hi = L - (n - 1 - i)
        fixed.append(max(lo, min(c, hi)))
    bounds = [0] + fixed + [L]
    return [clean[bounds[i]:bounds[i + 1]] for i in range(n)]


def build_timeline(chunks: list, groups: list, group_texts: list,
                   total_duration: float) -> tuple:
    """生成片段表 + 句子级字幕表。"""
    segs, srt = [], []
    for gi, idxs in enumerate(groups):
        txt = group_texts[gi] if gi < len(group_texts) else ""
        durs = [chunks[i][1] - chunks[i][0] for i in idxs]
        parts = assign_text(txt, len(idxs), durs)
        for k, i in enumerate(idxs):
            a, b = chunks[i]
            segs.append({"id": f"g{gi+1:02d}c{k+1:02d}", "group": gi,
                         "start": a, "end": b, "text": parts[k]})
        srt.append((chunks[idxs[0]][0], chunks[idxs[-1]][1], txt))
    return segs, srt
