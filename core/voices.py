"""音色库：edge-tts 在线神经语音，按男声/女声分组。

voice_id 约定： edge:<ShortName>
选到哪个就自动切到对应音色参数，用户不用关心底层。
想接入 CosyVoice / GPT-SoVITS 等自定义音色，见 docs/AI声音工具接入指南.md。
"""

EDGE_VOICES = [
    # —— 男声 ——
    ("zh-CN-YunxiNeural",    "male",   "云希 · 温和自然（推荐）"),
    ("zh-CN-YunjianNeural",  "male",   "云健 · 沉稳浑厚（纪录片感）"),
    ("zh-CN-YunyangNeural",  "male",   "云扬 · 新闻播报（字正腔圆）"),
    ("zh-CN-YunxiaNeural",   "male",   "云夏 · 少年清亮"),
    ("zh-TW-YunJheNeural",   "male",   "雲哲 · 台湾男声"),
    ("zh-HK-WanLungNeural",  "male",   "瀚龍 · 粤语男声"),
    # —— 女声 ——
    ("zh-CN-XiaoxiaoNeural", "female", "晓晓 · 温柔知性（推荐）"),
    ("zh-CN-XiaoyiNeural",   "female", "晓伊 · 甜美亲和"),
    ("zh-CN-liaoning-XiaobeiNeural", "female", "晓北 · 东北女声"),
    ("zh-CN-shaanxi-XiaoniNeural",  "female", "晓妮 · 陕西女声"),
    ("zh-TW-HsiaoChenNeural", "female", "曉臻 · 台湾女声"),
    ("zh-TW-HsiaoYuNeural",   "female", "曉雨 · 台湾女声（柔和）"),
    ("zh-HK-HiuGaaiNeural",   "female", "曉佳 · 粤语女声"),
    ("zh-HK-HiuMaanNeural",   "female", "曉曼 · 粤语女声（沉稳）"),
]


def library():
    items = []
    for name, gender, label in EDGE_VOICES:
        items.append({"id": "edge:" + name, "engine": "edge-tts", "gender": gender,
                      "label": label, "group": "在线神经语音 · 男声" if gender == "male"
                      else "在线神经语音 · 女声"})
    # 本地大模型音色：需要先启动对应服务（见 docs/AI声音工具接入指南.md）
    items.append({"id": "cosyvoice:local", "engine": "cosyvoice", "gender": "female",
                  "label": "CosyVoice3 · 本地真人级（需启动服务）",
                  "group": "本地大模型 · 需启动服务"})
    return items


def grouped():
    out = {}
    for it in library():
        out.setdefault(it["group"], []).append(it)
    return out


def apply(cfg, voice_id):
    """把 voice_id 展开成引擎 + 音色参数。"""
    cfg = dict(cfg)
    if voice_id.startswith("edge:"):
        cfg["tts_engine"] = "edge-tts"
        cfg["edge_voice"] = voice_id[5:]
    elif voice_id.startswith("cosyvoice:"):
        # prompt_wav/prompt_text 等参数从 config.json 的 cosy_* 字段读取
        cfg["tts_engine"] = "cosyvoice"
    # 自定义引擎（sovits: 等）在此按前缀扩展，
    # 见 docs/AI声音工具接入指南.md
    cfg["voice"] = voice_id
    return cfg


def default_id():
    return "edge:zh-CN-YunxiNeural"
