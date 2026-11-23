"""扫描本地 VITS 的 174 个说话人，按基频分出男声/女声，生成 voices_local.json。

软件里点「刷新本地音色」会重新跑这个脚本；首次已预生成好，直接读缓存即可。
"""
import os
import json
import sys
import numpy as np
import librosa
import sherpa_onnx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "voices_local.json")
MODEL_DIR = os.path.join(ROOT, "models", "aishell3")
PROBE = "请大家到室外专用充电棚充电，守护全家平安。"


def build_tts():
    vits = sherpa_onnx.OfflineTtsVitsModelConfig(
        model=os.path.join(MODEL_DIR, "vits-aishell3.onnx"),
        lexicon=os.path.join(MODEL_DIR, "lexicon.txt"),
        tokens=os.path.join(MODEL_DIR, "tokens.txt"))
    cfg = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(vits=vits, num_threads=4),
        max_num_sentences=1)
    return sherpa_onnx.OfflineTts(cfg)


def f0_of(y, sr):
    y = y[np.abs(y) > 0.002]
    if len(y) < 2048:
        return 0.0, 0.0
    f = librosa.yin(y.astype(np.float32), fmin=70, fmax=330,
                    sr=sr, frame_length=1024)
    v = f[(f > 70) & (f < 330)]
    return (float(np.median(v)) if len(v) else 0.0), float(len(v) / max(len(f), 1))


def style(gender, f0):
    if gender == "male":
        for lim, name in [(115, "低沉"), (130, "沉稳"), (145, "温和"), (160, "清亮")]:
            if f0 < lim:
                return name
        return "清亮"
    for lim, name in [(190, "温婉"), (215, "甜美"), (240, "清脆"), (400, "明亮")]:
        if f0 < lim:
            return name
    return "明亮"


def main():
    tts = build_tts()
    sr, n = tts.sample_rate, getattr(tts, "num_speakers", 174)
    rows = []
    for sid in range(n):
        try:
            a = tts.generate(PROBE, sid=sid, speed=1.0)
            y = np.asarray(a.samples, dtype=np.float32)
            f0, ratio = f0_of(y, sr)
            gender = "male" if 90 <= f0 <= 160 else ("female" if 165 <= f0 <= 280 else "")
            if gender and ratio > 0.55:
                rows.append({"sid": sid, "gender": gender, "f0": round(f0, 1),
                             "ratio": round(ratio, 2), "style": style(gender, f0)})
        except Exception:
            continue
        if sid % 30 == 0:
            print(f"  scanned {sid}/{n}", flush=True)

    # 每类挑 10 个：基频分布尽量铺开，避免都挤在一个音区
    picked = []
    for g in ("male", "female"):
        pool = sorted([r for r in rows if r["gender"] == g], key=lambda r: r["f0"])
        if not pool:
            continue
        step = max(1, len(pool) // 10)
        chosen = pool[::step][:10]
        if pool[-1] not in chosen:
            chosen.append(pool[-1])
        for r in chosen:
            r["label"] = f"本地{r['style']}{'男' if g=='male' else '女'}声 · {r['sid']}号 ({r['f0']:.0f}Hz)"
            picked.append(r)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
    print(f"DONE male={sum(1 for r in picked if r['gender']=='male')} "
          f"female={sum(1 for r in picked if r['gender']=='female')} -> {OUT}")


if __name__ == "__main__":
    main()
