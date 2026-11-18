"""软件配置：默认值 + 持久化。所有参数用户都能在界面里改，改完写回 config.json。

开源便携版：不内置任何模型、不写死任何本机路径。
ffmpeg 默认从 PATH 查找；台词来源默认 manual（手动/粘贴），
想要离线语音识别请自行下载 SenseVoice 模型，见 README。
"""
import os
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.json")

DEFAULT = {
    "input": "",
    "output": "",
    "work_dir": os.path.join(ROOT, "work"),
    "out_dir": os.path.join(ROOT, "输出"),

    "tts_engine": "edge-tts",              # edge-tts | 自定义引擎（见 docs/）
    "voice": "edge:zh-CN-YunxiNeural",      # 统一音色 ID：edge:<名>
    "edge_voice": "zh-CN-YunxiNeural",      # 温和男声
    "edge_rate": "-4%",
    "edge_pitch": "-2Hz",
    "edge_concurrency": 4,

    "asr_engine": "manual",                # manual（默认） | srt | sensevoice（需自行下模型）
    "speaker_gap": 0.60,                   # 静音 >= 该值视为句子分界（秒）
    "min_silence": 0.20,                   # 短于此的静音归入发声段（秒）
    "noise_db": -38,                       # 静音判定阈值（dBFS）
    "adaptive_ratio": 0.04,                # 自适应阈值 = 95 分位能量 × 该比例
    "pace_balance": True,                  # 语速均衡：消除忽快忽慢（强烈建议开）
    "pace_borrow": 0.90,                   # 单段最多向后借用多少秒静音
    "pace_pause_mid": 0.18,               # 句内每段目标停顿（秒）：避免零间隔挤成一团
    "pace_pause_bnd": 0.55,               # 句尾目标停顿（秒）：自然句间留白

    "tone_match": True,                    # 音色一致：逐段频谱匹配到全片参考谱
    "tone_strength": 0.85,                 # 匹配强度 0~1（1 为完全拉齐，限幅 ±4dB）
    "loudness_match": True,                # 逐段 RMS 对齐，避免一句响一句闷
    "output_peak": 0.95,                   # 整轨峰值（实测 AAC 192k 无过冲，可贴近原片响度）
    "allow_stretch": False,                # 是否允许相位声码器变速（开启会发蒙）
    "rate_iterations": 2,                  # TTS 语速参数精调轮数
    "rate_headroom": 0.985,                # 目标时长余量，宁可尾部补静音也不吞字
    "tts_cache": True,                     # 缓存合成结果：锁定随机性、断网可复现、重跑更快

    # 可选：离线语音识别模型（不随软件分发，需自行下载放到该目录）
    "sensevoice_dir": os.path.join(ROOT, "models", "sensevoice"),
    "ffmpeg": "ffmpeg",                    # 默认从 PATH 查找，也可填绝对路径
    "sample_rate": 48000,
    "video_bitrate": "192k",
}


def load() -> dict:
    cfg = dict(DEFAULT)
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                cfg.update(json.load(f))
        except Exception:
            pass
    return cfg


def save(cfg: dict) -> None:
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def resolve_ffmpeg(cfg: dict) -> str:
    p = cfg.get("ffmpeg", "")
    if p and os.path.exists(p):
        return p
    from shutil import which
    return which("ffmpeg") or p or "ffmpeg"
