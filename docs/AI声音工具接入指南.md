# 接入你自己的 AI 声音工具

DubTool 最值钱的部分是**时间轴对齐**（波形切分、语速均衡、采样点级贴合、频谱/响度匹配），这部分和「用什么合成语音」完全解耦。所以接入 CosyVoice、GPT-SoVITS、Fish-Speech 或任何自研 TTS，**只动一个文件：`core/tts.py`**，其余全部原样复用。

---

## 一、引擎接口契约

`core/pipeline.py` 只要求合成引擎实现三个方法：

```python
class YourEngine:
    name = "your-engine"

    def synth_batch(self, texts, log=print):
        """一次性合成整批。
        texts: [str, ...]        每段要读的文字
        返回:  [(wave, sr), ...] 与输入等长的列表，wave 是 float32 一维 numpy 数组
        """
        ...

    def resynth(self, index, text, speed):
        """重合成单段（试听/微调用）。返回 (wave, sr)。"""
        ...

    def resynth_many(self, jobs, texts, tag="r"):
        """批量按语速重合成，用于语速精调。
        jobs: [(index, pct), ...]   pct 是相对语速百分比，如 -20 表示慢 20%
        返回: {index: (wave, sr)}
        """
        ...
```

**采样率随意**——pipeline 后续会统一重采样到 `config.json` 的 `sample_rate`（默认 48000）。wave 务必是 `float32`、`[-1, 1]` 范围、一维。

---

## 二、最小接法（3 步）

### 第 1 步：在 `core/voices.py` 加音色

```python
EDGE_VOICES = [ ... ]   # 保留原有

CUSTOM_VOICES = [
    ("关中女声", "female", "CosyVoice · 关中女声（自克隆）"),
]

def library():
    items = []
    # ... 原有 edge 音色 ...
    for name, gender, label in CUSTOM_VOICES:
        items.append({"id": "cosyvoice:" + name, "engine": "cosyvoice",
                      "gender": gender, "label": label,
                      "group": "自定义 AI 声音"})
    return items

def apply(cfg, voice_id):
    cfg = dict(cfg)
    if voice_id.startswith("edge:"):
        cfg["tts_engine"] = "edge-tts"
        cfg["edge_voice"] = voice_id[5:]
    elif voice_id.startswith("cosyvoice:"):
        cfg["tts_engine"] = "cosyvoice"
        cfg["cosy_voice"] = voice_id[10:]
    cfg["voice"] = voice_id
    return cfg
```

### 第 2 步：在 `core/tts.py` 写引擎类

以 CosyVoice 为例（假设你已按官方方式起了本地推理服务/或在进程内加载了模型）：

```python
class CosyVoiceEngine:
    name = "cosyvoice"

    def __init__(self, voice, cache=None, **kw):
        import requests                      # 或直接在进程内 import cosyvoice
        self.endpoint = kw.get("endpoint", "http://127.0.0.1:50000/tts")
        self.voice = voice
        self.cache = cache
        self.sr = 24000                      # CosyVoice 常见输出采样率，按实际改

    def _synth(self, text, speed=1.0):
        # 关键：把 speed 换算成引擎认识的参数。CosyVoice 多用 instruct 文本控制，
        # 例如「语速稍慢」，也可映射到它支持的 rate 字段。
        r = requests.post(self.endpoint, json={
            "text": text,
            "speaker": self.voice,
            "speed": speed,
        }, timeout=60)
        r.raise_for_status()
        # 假设返回 16-bit PCM，采样率 self.sr
        import numpy as np
        y = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
        return np.ascontiguousarray(y, dtype=np.float32)

    def _k(self, text, speed):
        from .tts_cache import key
        return key("cosyvoice", self.voice, f"{float(speed):.3f}", text)

    def _gen(self, text, speed=1.0):
        # 缓存命中直接复用：这是「抽卡类」模型必须做的，否则每次重跑都变声
        if self.cache is not None:
            hit = self.cache.get(self._k(text, speed))
            if hit is not None:
                y, sr = hit
                return y, sr
        y = self._synth(text, speed)
        if self.cache is not None:
            self.cache.put(self._k(text, speed), y, self.sr)
        return y, self.sr

    def synth_batch(self, texts, log=print):
        out = [self._gen(t or " ", 1.0) for t in texts]
        return out

    def resynth(self, index, text, speed):
        return self._gen(text, float(np.clip(speed, 0.5, 2.0)))

    def resynth_many(self, jobs, texts, tag="r"):
        out = {}
        for i, pct in jobs:
            out[i] = self._gen(texts[i], float(np.clip(1.0 + pct/100, 0.5, 2.0)))
        return out
```

### 第 3 步：在 `make_engine()` 加一行分支

```python
def make_engine(cfg, ffmpeg):
    cache = None
    if cfg.get("tts_cache", True):
        root = cfg.get("tts_cache_dir") or os.path.join(cfg["work_dir"], "tts_cache")
        cache = DiskCache(root)
    if cfg["tts_engine"] == "cosyvoice":
        return CosyVoiceEngine(cfg["cosy_voice"], cache=cache,
                               endpoint=cfg.get("cosy_endpoint"))
    return EdgeTTSEngine(...)   # 默认
```

完事。界面选「自定义 AI 声音」里的音色即可，时间轴、语速均衡、频谱匹配、缓存锁定全部照常工作。

---

## 三、接入要点（踩坑提醒）

### 1. 抽卡类模型一定要用缓存
CosyVoice / GPT-SoVITS / VITS 这类生成式声学模型，同一句话每次采样都不同（实测本地 VITS 8 次长度差 17.7%）。DubTool 的 `tts_cache` 会按「引擎+音色+语速+文本」算 SHA-256 存波形，**抽一次锁死**，之后命中直接复用。所以：
- 保持「锁定合成结果」勾选；
- 引擎的 `_k()` 必须把**所有影响输出的参数**纳入 key，否则会拿旧声音凑数。

### 2. 语速映射
pipeline 的语速精调靠 `native(pct) ≈ K/(1+pct/100)` 这个近似，前提是「语速参数能连续、单调地改变时长」。edge-tts 的 `rate` 满足；CosyVoice 如果用 instruct 文本控制语速（「慢一点」），这个精调就不准。对策二选一：
- 优先用引擎的**数值型 speed/rate 字段**（如果有）；
- 没有的话，在 `resynth_many` 里把 pct 量化到几档 instruct（如 -30→「很慢」、0→「正常」、+30→「快一点」），并在 `_k` 里带上档位，精调轮数设 1。

### 3. 采样率与波形范围
不同引擎输出采样率不同（CosyVoice 24k、GPT-SoVITS 32k 等），`synth_batch` 返回真实的 `sr`，pipeline 会重采样。但务必保证：
- 波形是 `float32` 一维；
- 幅度在 `[-1, 1]`，别返回 int16 原样；
- 首尾别带大段静音，否则 `align.voiced_seconds` 测自然时长会偏。

### 4. 长文本与并发
pipeline 是**逐窗口**调用合成的（每段一句话），不是一次扔整篇。所以：
- 引擎按「单句」设计即可，不用处理长文本分段；
- `synth_batch` 里若引擎支持并发，可以自己起线程池；不支持就串行，慢一点但稳。

### 5. 想让新引擎在界面可调参数
在 `config.json` 加字段（如 `cosy_endpoint`），界面里加输入框，或干脆手动改 json——`config.load()` 会把 json 里的键合并进默认配置，引擎构造时从 `cfg` 读即可。

---

## 四、三个引擎的接入路线对比

| 引擎 | 路线 | 你要准备 | 抽卡 | 适合 |
|---|---|---|---|---|
| **CosyVoice 3.0 / 2.0** | 指令直出 / 声音克隆 | 一键整合包（约 6~12GB），N 卡 6~8G 显存 | 是（靠缓存锁） | 普通话拟真、方言、语气控制，**最贴合本项目** |
| **GPT-SoVITS** | 克隆真人 | 1 分钟干净人声样本 + 微调 | 是（靠缓存锁） | 要复刻某个特定的人的声音 |
| **Fish-Speech** | 零样本克隆 | 5~10 秒样本 | 是（靠缓存锁） | 快速克隆、稳定性好 |
| edge-tts（自带） | 云端 | 联网即可 | 否（确定性） | 开箱即用、零成本、音质已够好 |

