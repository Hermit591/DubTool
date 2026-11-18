"""完整链路编排：波形切时间轴 → 文字识别 → TTS → 采样点级对齐 → 封装视频。

不依赖任何智能体。默认链路：台词手动输入 / 导入 SRT + edge-tts 合成，
只要有 ffmpeg 就能跑；识别与合成引擎都可替换，见 docs/AI声音工具接入指南.md。
"""
import os
import json
import shutil
import numpy as np

from . import extract, vad, asr, tts, align, mux, voices, pace, pron
from .config import resolve_ffmpeg


class Cancelled(Exception):
    pass


def _default_output(cfg):
    if cfg.get("output"):
        return cfg["output"]
    stem = os.path.splitext(os.path.basename(cfg["input"]))[0]
    return os.path.join(cfg["out_dir"], f"{stem}_配音.mp4")


def run(cfg, log=print, progress=lambda pct, msg: None, cancel=lambda: False):
    cfg = voices.apply(cfg, cfg.get("voice") or voices.default_id())
    ffmpeg = resolve_ffmpeg(cfg)
    video = cfg["input"]
    if not video or not os.path.exists(video):
        raise FileNotFoundError("请先选择要配音的视频/音频文件")
    work = cfg["work_dir"]
    os.makedirs(work, exist_ok=True)
    sr_out = int(cfg.get("sample_rate", 48000))

    def ck(pct, msg):
        if cancel():
            raise Cancelled()
        log(msg)
        progress(pct, msg)

    ck(3, "[1/6] 提取音轨…")
    y, sr, audio_dur, _ = extract.extract(video, work, ffmpeg)
    vdur = extract.probe_duration(video, ffmpeg)
    total = vdur if vdur > 0 else audio_dur
    ck(8, f"     音轨 {audio_dur:.3f}s / 视频 {total:.3f}s")

    group_texts = []
    tf = cfg.get("timeline_file", "")
    if tf and os.path.exists(tf):
        with open(tf, "r", encoding="utf-8") as f:
            saved = json.load(f)
        segs = saved["segments"]
        total = float(saved.get("duration", total))
        srt = [tuple(x) for x in saved.get("srt", [])]
        group_texts = [t for _, _, t in srt]
        ck(35, f"[2/6] 复用时间轴 {tf}，{len(segs)} 个片段（文字可直接改）")
    else:
        ck(12, "[2/6] 波形静音分析，切出发声窗口…")
        chunks = vad.detect_chunks(y, sr, cfg["noise_db"],
                                   cfg["adaptive_ratio"], cfg["min_silence"])
        groups = vad.group_sentences(chunks, cfg["speaker_gap"])
        if not chunks:
            raise RuntimeError("没检测到人声，检查素材或调低「静音阈值」")
        ck(18, f"     {len(chunks)} 个发声窗口 / {len(groups)} 个句子组")

        ck(22, f"[3/6] 台词识别（{cfg['asr_engine']}）…")
        group_texts = asr.transcribe_groups(y, sr, chunks, groups,
                                            cfg["asr_engine"], cfg, log=log)
        if cancel():
            raise Cancelled()
        segs, srt = vad.build_timeline(chunks, groups, group_texts, total)
        ck(35, f"     台词：{' / '.join(t[:14] for t in group_texts)}")

        with open(os.path.join(work, "timeline.json"), "w", encoding="utf-8") as f:
            json.dump({"duration": total, "segments": segs, "srt": srt}, f,
                      ensure_ascii=False, indent=2)

    # 语速均衡：把「整句文字机械按窗口数切」和「波形强制铺满窗口」这两个
    # 造成忽快忽慢的环节一并修正，重算文字切点与每段实际发声终点。
    pace_stats = []
    if cfg.get("pace_balance", True):
        segs, pace_stats = pace.balance(
            segs, srt, borrow=float(cfg.get("pace_borrow", 0.90)), log=log,
            target_mid=float(cfg.get("pace_pause_mid", 0.18)),
            target_bnd=float(cfg.get("pace_pause_bnd", 0.55)))
        ck(38, f"     语速均衡完成，{len(pace_stats)} 段重排")
    else:
        # 完全手动模式：尊重用户在 JSON 里写好的 start/end/text/voice_end，
        # 不再重切文字、也不删除 voice_end（缺字段时退回 end）。
        for s in segs:
            if not s.get("voice_end"):
                s["voice_end"] = s["end"]

    ck(40, f"[4/6] 合成语音（{cfg['tts_engine']}）…")
    engine = tts.make_engine(cfg, ffmpeg)
    # 多音字校准：合成用「读音文本」，字幕/时间轴仍用原字
    pron.reload()
    texts = [pron.fix(s["text"]) for s in segs]
    n_fix = sum(1 for s, t in zip(segs, texts) if (s["text"] or "") != t)
    if n_fix:
        log(f"     多音字校准：{n_fix} 段已按读音替换后合成（字幕保留原字）")
    audios = engine.synth_batch(texts, log=log)
    if cancel():
        raise Cancelled()

    # 在 TTS 参数域把每段自然时长调准 —— 波形不再经过相位声码器。
    # 实测 native(pct) ≈ K/(1+pct/100)，反解一次即可把误差收敛到 ±3%，
    # 剩下的一点差值用补静音/裁剪消化，音质全程保持原生。
    base_pct = 0.0
    if cfg["tts_engine"] == "edge-tts":
        base_pct = float(str(cfg.get("edge_rate", "-4%")).replace("%", "").replace("+", "") or 0)
    pcts = [base_pct for _ in segs]
    headroom = float(cfg.get("rate_headroom", 0.985))
    iterations = max(1, int(cfg.get("rate_iterations", 2)))
    for rnd in range(iterations):
        jobs = []
        for i, seg in enumerate(segs):
            target = float(seg.get("voice_end", seg["end"])) - seg["start"]
            if target <= 0.2 or not seg["text"].strip():
                continue
            wav, s = audios[i]
            native = align.voiced_seconds(wav, s)
            if native <= 0.05:
                continue
            K = native * (1.0 + pcts[i] / 100.0)
            new_pct = float(np.clip(100.0 * (K / (target * headroom) - 1.0), -45.0, 75.0))
            if abs(new_pct - pcts[i]) >= 2.0:
                jobs.append((i, new_pct))
            pcts[i] = new_pct
        if not jobs:
            break
        try:
            for i, item in engine.resynth_many(jobs, texts, tag=f"r{rnd}").items():
                audios[i] = item
        except Exception as e:
            log(f"     第 {rnd+1} 轮语速精调跳过（{type(e).__name__}），保留首次合成")
            break
    retune = []
    for i, seg in enumerate(segs):
        target = float(seg.get("voice_end", seg["end"])) - seg["start"]
        wav, s = audios[i]
        native = align.voiced_seconds(wav, s)
        retune.append({"id": seg["id"], "rate_pct": round(pcts[i], 1),
                       "native": round(native, 3), "target": round(target, 3),
                       "fit": round(native / target, 3) if target > 0 else 0.0})
    fitv = np.array([r["fit"] for r in retune if r["target"] > 0.2])
    ck(62, f"     语速精调：rate {min(pcts):+.0f}% ~ {max(pcts):+.0f}%，"
           f"自然/目标 {fitv.min():.2f} ~ {fitv.max():.2f}")

    # 字幕/时间轴只展示原字：去掉行内读音标注（合成已用 fix 后的文本）
    for s in segs:
        s["text"] = pron.clean(s["text"])
    srt = [(a, b, pron.clean(t)) for a, b, t in srt]

    ck(70, "[5/6] 采样点级对齐 + 音色一致化…")
    track, report = align.build_track(
        segs, audios, total, sr_out, log=log,
        tone_match=bool(cfg.get("tone_match", True)),
        tone_strength=float(cfg.get("tone_strength", 0.85)),
        loudness_match=bool(cfg.get("loudness_match", True)),
        output_peak=float(cfg.get("output_peak", 0.89)),
        allow_stretch=bool(cfg.get("allow_stretch", False)))
    wav_path = os.path.join(work, "dub_48k.wav")
    mux.write_wav(track, sr_out, wav_path)

    ck(85, "[6/6] 封装视频与校验…")
    out = _default_output(cfg)
    if mux.has_video(video, ffmpeg):
        mux.mux(video, wav_path, out, ffmpeg, cfg.get("video_bitrate", "192k"))
    else:
        out = os.path.splitext(out)[0] + ".wav"
        shutil.copyfile(wav_path, out)
        log("     输入没有视频流，直接交付配音音轨")
    stem = os.path.splitext(os.path.basename(out))[0]
    base = os.path.dirname(out)
    srt_path = mux.write_srt(srt, os.path.join(base, stem + "_台词.srt"))
    verify = mux.verify(track, sr_out, segs, total)
    mux.write_report({"tts_engine": cfg["tts_engine"], "asr_engine": cfg["asr_engine"],
                      "pace_balance": bool(cfg.get("pace_balance", True)),
                      "tone_match": bool(cfg.get("tone_match", True)),
                      "tone_strength": float(cfg.get("tone_strength", 0.85)),
                      "pace": pace_stats, "retune": retune,
                      "verify": verify, "segments": report},
                     os.path.join(base, stem + "_校验报告.json"))
    with open(os.path.join(base, stem + "_时间轴.json"), "w", encoding="utf-8") as f:
        json.dump({"duration": total, "segments": segs, "srt": srt}, f,
                  ensure_ascii=False, indent=2)

    # 顺手导出试听 mp3
    try:
        import subprocess
        mp3 = os.path.join(base, stem + "_试听.mp3")
        subprocess.run([ffmpeg, "-y", "-v", "error", "-i", wav_path,
                        "-codec:a", "libmp3lame", "-b:a", "192k", mp3], check=True)
    except Exception:
        mp3 = ""
        log("     MP3 试听导出跳过（不影响成品）")

    ck(100, f"完成：{out}")
    return {"output": out, "wav": wav_path, "srt": srt_path, "mp3": mp3,
            "verify": verify, "segments": segs, "group_texts": group_texts,
            "timeline": os.path.join(work, "timeline.json")}
