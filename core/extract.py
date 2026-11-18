"""第 1 步：从视频里抽出 16k 单声道音轨（配音时间轴的唯一依据）。"""
import os
import subprocess
import numpy as np
import soundfile as sf


def extract(video: str, work_dir: str, ffmpeg: str) -> tuple:
    """返回 (波形 float32, 采样率 16000, 时长秒, wav 路径)。"""
    os.makedirs(work_dir, exist_ok=True)
    wav = os.path.join(work_dir, "orig16.wav")
    cmd = [ffmpeg, "-y", "-v", "error", "-i", video,
           "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", wav]
    subprocess.run(cmd, check=True)
    y, sr = sf.read(wav, dtype="float32")
    return y, sr, len(y) / sr, wav


def probe_duration(video: str, ffmpeg: str) -> float:
    """用 ffprobe 拿容器时长（视频可能比音轨长，最终音轨按视频时长铺）。"""
    ffprobe = ffmpeg.replace("ffmpeg.exe", "ffprobe.exe").replace("ffmpeg", "ffprobe")
    try:
        out = subprocess.check_output(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", video], text=True, timeout=20)
        return float(out.strip())
    except Exception:
        return 0.0
