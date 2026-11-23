"""完整链路编排：波形切时间轴 → 文字识别 → TTS → 采样点级对齐 → 封装视频。

任何一步都不依赖 WorkBuddy 或云端 ASR，装好依赖后可完全离线运行
（选 local-vits + sensevoice 时；edge-tts 需要联网但不需要智能体）。
"""
import os
import json
import shutil
import numpy as np

from . import extract, vad, asr, tts, align, mux, voices, pace, pron, sentence
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

    # 节奏策略：
    #   · 复用了时间轴 = 用户手改过的文件，完全以它为准（text/start/end/voice_end
    #     原样生效），不再自动重排，也不需要任何 locked 标记。
    #   · 只有全新识别（没有时间轴）才跑 pace.balance 自动均衡，修忽快忽慢/贴脸段。
    reuse_timeline = bool(tf and os.path.exists(tf))
    pace_stats = []
    if cfg.get("pace_balance", True) and not reuse_timeline:
        segs, pace_stats = pace.balance(
            segs, srt, borrow=float(cfg.get("pace_borrow", 0.90)), log=log,
            target_mid=float(cfg.get("pace_pause_mid", 0.18)),
            target_bnd=float(cfg.get("pace_pause_bnd", 0.55)))
        ck(38, f"     语速均衡完成，{len(pace_stats)} 段重排")
    else:
        # 尊重 JSON 里写好的 start/end/text/voice_end；缺 voice_end 时退回 end。
        for s in segs:
            if not s.get("voice_end"):
                s["voice_end"] = s["end"]
        if reuse_timeline:
            ck(38, "[时间轴] 已完全以你改过的文件为准（不自动重排、无需 locked）")

    ck(40, f"[4/6] 合成语音（{cfg['tts_engine']}）…")
    engine = tts.make_engine(cfg, ffmpeg)
    # 多音字校准：合成用「读音文本」，字幕/时间轴仍用原字
    pron.reload()
    texts = [pron.fix(s["text"]) for s in segs]
    n_fix = sum(1 for s, t in zip(segs, texts) if (s["text"] or "") != t)
    if n_fix:
        log(f"     多音字校准：{n_fix} 段已按读音替换后合成（字幕保留原字）")
    # ---- 波形定长 + 弹性停顿：按句子整段合成，句内绝不变速 ----
    # 旧策略逐段拧 rate（-14%~+37%）把波形硬塞进原窗口，是「时快时慢」的根源。
    # 新策略：一句一次合成，用真实波形时长定发声终点；超出到下一句的空间时
    # 整句统一微调一次（限幅），句内同速；句尾停顿由原声间隔弹性吸收。
    groups, order = sentence.group_indices(segs)
    sent_texts = ["".join(texts[i] for i in groups[g]) for g in order]
    if cancel():
        raise Cancelled()
    sent_audios = engine.synth_batch(sent_texts, log=log)
    if cancel():
        raise Cancelled()
    audios, retune = sentence.plan(
        segs, order, groups, sent_audios, sent_texts, engine, total, log=log,
        max_adjust=float(cfg.get("sentence_max_adjust", 0.15)))
    ck(62, f"     句子级定长完成：{len(retune)} 句，句内同速")

    # 字幕/时间轴只展示原字：去掉行内读音标注（合成已用 fix 后的文本）
    for s in segs:
        s["text"] = pron.clean(s["text"])
    # 字幕终点同步到新的发声终点：起点锚定画面切换点不变，终点跟着波形走，
    # 避免声音念完了字幕还挂着（或字幕先走声音还没到）的错位感。
    srt_new = []
    for k, (a, b, t) in enumerate(srt):
        if k < len(order):
            last = segs[groups[order[k]][-1]]
            b = float(last.get("voice_end", b))
        srt_new.append((a, round(b, 3), pron.clean(t)))
    srt = srt_new

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
