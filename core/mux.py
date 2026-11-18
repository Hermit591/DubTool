"""第 6 步：落盘音轨、封装视频、写字幕与对齐校验报告。"""
import os
import json
import subprocess
import numpy as np
import soundfile as sf


def ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_wav(track, sr, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stereo = np.column_stack([track, track]).astype(np.float32)
    sf.write(path, stereo, sr, subtype="PCM_16")
    return path


def has_video(path, ffmpeg):
    """纯音频输入时不要去 map 视频流，直接交付配音音轨。"""
    ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=index", "-of", "csv=p=0", path],
            text=True, timeout=20)
        return bool(out.strip())
    except Exception:
        return False


def mux(video, wav, out, ffmpeg, bitrate="192k"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", video, "-i", wav,
           "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
           "-c:a", "aac", "-b:a", bitrate, "-ar", "48000", "-shortest", out]
    subprocess.run(cmd, check=True)
    return out


def write_srt(srt_items, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = "\n".join(
        f"{i}\n{ts(a)} --> {ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(srt_items, 1))
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)
    return path


def verify(track, sr, segments, video_duration):
    """返回可写进报告的校验数据：时长差、发声区间内能量、真实静音段能量。

    语速均衡后，一段语音可能越过窗口尾部借用静音（voice_end > end），
    此时「窗口起点前」不再是静音，不能拿它判异常。改用两段 voice_end
    与下一段 start 之间的真实空白做检测。
    """
    rows = []
    for i, s in enumerate(segments):
        a = int(s["start"] * sr)
        b = int(s.get("voice_end", s["end"]) * sr)
        inside = track[a:min(b, len(track))]
        prev_end = float(segments[i - 1].get("voice_end", segments[i - 1]["end"])) if i else 0.0
        gap_a = int(prev_end * sr)
        gap = track[gap_a:a] if a > gap_a else np.zeros(1, dtype=np.float32)
        rows.append({
            "id": s["id"],
            "start": s["start"], "end": s["end"],
            "voice_end": round(float(s.get("voice_end", s["end"])), 4),
            "text": s["text"],
            "dub_rms": round(float(np.sqrt(np.mean(inside ** 2))), 5) if len(inside) else 0.0,
            "gap_rms": round(float(np.sqrt(np.mean(gap ** 2))), 6) if len(gap) else 0.0,
        })
    # 空文本窗口（如 ASR 误检到的空句）本就该是静音，不计入异常
    bad = [r for r in rows if (r["dub_rms"] < 0.01 or r["gap_rms"] > 0.01)
           and (r["text"] or "").strip()]
    return {
        "duration_video": round(video_duration, 4),
        "duration_track": round(len(track) / sr, 4),
        "drift_ms": round((len(track) / sr - video_duration) * 1000, 3),
        "segments": len(rows), "bad_segments": len(bad), "rows": rows,
        "peak": round(float(np.max(np.abs(track))), 4),
    }


def write_report(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path
