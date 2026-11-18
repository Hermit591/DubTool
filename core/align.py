"""第 5 步：把每段 TTS 波形按采样点数精确压回原发声窗口。

音质一致性设计（v2）：
  * 波形一律不经过相位声码器 —— 时长由 TTS 参数在合成阶段调准，
    这里只做裁剪/补静音。相位声码器的 phasiness 是「蒙了一层」的主因，
    而旧版只有部分段触发它，于是逐段清晰度不一致。
  * 全片统一参考谱 + 逐段温和频谱匹配，把神经 TTS 因文本不同产生的
    高频占比波动拉齐（限制 ±4dB、1/3 倍频程平滑，不产生电话音）。
  * RMS 匹配替代峰值匹配，段落响度不再忽大忽小。
  * 去掉逐段 tanh 软限幅，只在整轨做一次线性峰值保护。
"""
from math import gcd
import numpy as np
import librosa
from scipy.signal import resample_poly, butter, sosfilt

N_FFT = 2048
HOP = 512


def resample(y, src_sr, dst_sr):
    if src_sr == dst_sr:
        return np.asarray(y, dtype=np.float32)
    g = gcd(int(src_sr), int(dst_sr))
    return resample_poly(y, dst_sr // g, src_sr // g).astype(np.float32)


def trim_silence(y, sr, top_db=26.0):
    """引擎自带的前后静音要去掉：停顿由时间轴负责，不能让 TTS 自己加。"""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < 512:
        return y
    yt, _ = librosa.effects.trim(y, top_db=top_db,
                                 frame_length=min(1024, len(y) // 4 or 256),
                                 hop_length=min(256, len(y) // 16 or 64))
    return yt if len(yt) > 256 else y


def fit_exact(y, sr, target_n, allow_stretch=False, max_stretch=1.12):
    """精确裁剪/补零到 target_n 个采样点；默认完全不变速。

    旧版在这里用相位声码器微调时长，结果只有 need 偏离 1 的段被处理，
    造成「有的清亮有的发蒙」。v2 把调速前移到 TTS 参数阶段，此处保持原生波形。
    allow_stretch=True 可回退旧行为。
    """
    y = np.asarray(y, dtype=np.float32)
    if len(y) == 0 or target_n <= 0:
        return np.zeros(max(target_n, 0), dtype=np.float32)
    if allow_stretch and len(y) > 2048:
        rate = float(np.clip(len(y) / target_n, 1 / max_stretch, max_stretch))
        if abs(rate - 1.0) > 0.004:
            y = librosa.effects.time_stretch(y, rate=rate)
    if len(y) > target_n:
        y = y[:target_n]
    elif len(y) < target_n:
        y = np.pad(y, (0, target_n - len(y)))
    f = max(1, int(sr * 0.012))
    if target_n > 2 * f:                                   # 12ms 淡入淡出，边界无 click
        w = np.linspace(0.0, 1.0, f, dtype=np.float32)
        y[:f] *= w
        y[-f:] *= w[::-1]
    return y.astype(np.float32)


def excite(y, sr, amount=0.16):
    """8kHz 本地模型的带宽补偿：用低通信号的谐波轻补 4~8kHz 清晰度。"""
    y = np.asarray(y, dtype=np.float32)
    if sr < 16000 or len(y) < 4096:
        return y
    lp = sosfilt(butter(2, 2800, btype="low", fs=sr, output="sos"), y).astype(np.float32)
    harm = np.tanh(lp * 2.6) * 0.34 - lp * 0.55
    harm = sosfilt(butter(2, 3900, btype="high", fs=sr, output="sos"), harm).astype(np.float32)
    return (y + amount * np.tanh(harm * 3.5) * 0.28).astype(np.float32)


def soften(y, sr, cutoff=11500.0):
    """透明柔化：只削超高频毛刺，不做非线性限幅。

    旧版的 np.tanh(y*1.10) 对已归一到 0.92 的波形产生谐波失真，
    实测把谱平坦度抬高 3~5 倍，听感就是「蒙了一层」。
    """
    y = np.asarray(y, dtype=np.float32)
    if sr / 2 > cutoff * 1.05:
        y = sosfilt(butter(2, min(cutoff, sr / 2 * 0.94),
                           btype="low", fs=sr, output="sos"), y).astype(np.float32)
    return y


def voiced_seconds(y, sr):
    return len(trim_silence(y, sr)) / sr


# ---------------------------------------------------------------- 频谱一致性

def avg_spectrum(y, sr, n_fft=N_FFT, hop=HOP):
    """只统计有声帧的平均幅度谱，静音帧不参与（否则会把增益曲线拉歪）。"""
    y = np.asarray(y, dtype=np.float32)
    if len(y) < n_fft * 2:
        return None
    w = np.hanning(n_fft).astype(np.float32)
    frames = np.stack([y[i:i + n_fft] * w for i in range(0, len(y) - n_fft, hop)])
    S = np.abs(np.fft.rfft(frames, axis=1))
    e = S.sum(axis=1)
    keep = e >= max(float(np.percentile(e, 30)), float(e.max()) * 0.05)
    S = S[keep].mean(axis=0) if int(keep.sum()) >= 2 else S.mean(axis=0)
    return S.astype(np.float64) + 1e-8


def tone_gain_db(seg_spec, ref_spec, sr, strength=0.85, max_db=4.0,
                 n_fft=N_FFT):
    """逐段增益曲线：把该段谱形状温和拉向全片参考谱。

    1/3 倍频程中位数平滑 → 不会跟着谐波梳状起伏；
    ±4dB 限幅 + 低端/高端淡出 → 不改变声线，只统一通透度。
    """
    f = np.fft.rfftfreq(n_fft, 1 / sr)
    raw = 20 * np.log10(np.clip(ref_spec / seg_spec, 0.1, 10.0))
    valid = (f >= 180) & (f <= 10000)
    out = np.zeros_like(raw)
    for i in np.flatnonzero(valid):
        lo, hi = f[i] * 2 ** (-1 / 6), f[i] * 2 ** (1 / 6)
        m = valid & (f >= lo) & (f <= hi)
        out[i] = float(np.median(raw[m])) if m.any() else 0.0
    out = np.clip(out * float(strength), -max_db, max_db)
    out *= np.clip((f - 120) / 80, 0, 1) * np.clip((12000 - f) / 2000, 0, 1)
    return f, out


def apply_gain(y, sr, freqs, gain_db):
    """频域乘平滑增益曲线；两端 reflect padding 消除循环卷积边界伪影。"""
    n = len(y)
    pad = int(sr * 0.04)
    z = np.pad(np.asarray(y, dtype=np.float32), (pad, pad), mode="reflect")
    Z = np.fft.rfft(z)
    fz = np.fft.rfftfreq(len(z), 1 / sr)
    g = np.interp(fz, freqs, gain_db, left=0.0, right=0.0)
    Z *= 10 ** (np.clip(g, -4.5, 4.5) / 20)
    return np.fft.irfft(Z, n=len(z))[pad:pad + n].astype(np.float32)


def spectral_metrics(y, sr):
    """给报告用的逐段清晰度指标。"""
    S = avg_spectrum(y, sr)
    if S is None:
        return {}
    f = np.fft.rfftfreq(N_FFT, 1 / sr)
    lo, hi = S[f < 3000].sum(), S[(f >= 3000) & (f < 9000)].sum()
    return {"centroid": round(float((f * S).sum() / S.sum()), 1),
            "hf_ratio": round(float(hi / max(lo, 1e-9)), 4),
            "flatness": round(float(np.exp(np.mean(np.log(S))) / np.mean(S)), 5),
            "rms": round(float(np.sqrt(np.mean(np.asarray(y) ** 2))), 4)}


# ---------------------------------------------------------------- 整轨构建

def build_track(segments, audios, total_duration, sr_out, log=print,
                tone_match=True, tone_strength=0.85, loudness_match=True,
                output_peak=0.95, allow_stretch=False):
    """audios: [(wave, sr), ...] 与 segments 一一对应。返回 48k 单声道整轨。"""
    track = np.zeros(int(round(total_duration * sr_out)), dtype=np.float32)

    # 第一遍：解码到统一采样率、去引擎静音、按需带宽补偿、透明柔化
    waves = []
    for y, sr_tts in audios:
        y = trim_silence(y, sr_tts)
        y = resample(y, sr_tts, sr_out)
        if sr_tts < 16000:
            y = excite(y, sr_out)
        waves.append(soften(y, sr_out))

    # 参考谱用中位数：某一段特别亮/特别闷时不会把整体目标带偏
    specs = [avg_spectrum(w, sr_out) for w in waves]
    good = [s for s in specs if s is not None]
    ref = np.median(np.stack(good), axis=0) if len(good) >= 3 else None

    before, after = [], []
    if tone_match and ref is not None:
        for i, w in enumerate(waves):
            if specs[i] is None:
                continue
            f, g = tone_gain_db(specs[i], ref, sr_out, strength=tone_strength)
            before.append(spectral_metrics(w, sr_out))
            waves[i] = apply_gain(w, sr_out, f, g)
            after.append(spectral_metrics(waves[i], sr_out))

    # RMS 匹配：峰值匹配会让擦音多的段落整体偏小声，听感上就成了「发闷」
    if loudness_match:
        rms = [float(np.sqrt(np.mean(w ** 2))) for w in waves]
        tgt = float(np.median([r for r in rms if r > 1e-5])) if any(r > 1e-5 for r in rms) else 0.0
        for i, w in enumerate(waves):
            if tgt > 0 and rms[i] > 1e-5:
                waves[i] = (w * float(np.clip(tgt / rms[i], 0.65, 1.55))).astype(np.float32)

    report = []
    for seg, y in zip(segments, waves):
        a = int(round(seg["start"] * sr_out))
        ve = float(seg.get("voice_end", seg["end"]))
        b = min(int(round(ve * sr_out)), len(track))
        b = max(b, a)
        track[a:b] += fit_exact(y, sr_out, b - a, allow_stretch=allow_stretch)
        m = spectral_metrics(y, sr_out)
        report.append({"id": seg["id"], "start": seg["start"], "end": seg["end"],
                       "text": seg["text"], "target": round((b - a) / sr_out, 4),
                       "samples": b - a,
                       "native": round(len(y) / sr_out, 4), "tone": m})

    # RMS 已逐段拉齐，整轨等比归一到目标峰值即可；AAC 编码有过冲，留足余量
    peak = float(np.max(np.abs(track)))
    if peak > 1e-6:
        track *= float(output_peak) / peak

    if log and before and after:
        b = np.array([x["hf_ratio"] for x in before])
        a2 = np.array([x["hf_ratio"] for x in after])
        spread = lambda v: (v.max() - v.min()) / max(v.mean(), 1e-9) * 100
        log(f"     音色一致：高频比离散 {spread(b):.0f}% → {spread(a2):.0f}%"
            f"（强度 {tone_strength}，峰值 {float(np.max(np.abs(track))):.3f}）")
    return track, report
