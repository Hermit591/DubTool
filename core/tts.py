"""逐窗口合成语音。

默认且唯一的内置引擎是 edge-tts（微软神经语音）：联网、快、音质好，
服务端逐样本确定性，同一输入重复合成结果完全一致。

想要 CosyVoice / GPT-SoVITS / 其他本地或自研引擎？本类就是可插拔接口：
任何引擎只要实现
    synth_batch(texts, log) -> [(波形 float32, 采样率), ...]
    resynth(index, text, speed) -> (波形 float32, 采样率)
    resynth_many(jobs, texts, tag) -> {index: (波形, 采样率)}
三个方法即可无缝接入 pipeline，时间轴对齐那套完全不用动。
详见 docs/AI声音工具接入指南.md。

所有引擎共享 tts_cache：按（引擎 / 音色 / 参数 / 文本）做 key，
第一次生成后落盘，之后命中即复用，成品逐样本可复现、断网也能复跑。
"""
import os
import asyncio
import subprocess
import numpy as np

from .tts_cache import DiskCache, key as cache_key

SR = 48000


def _decode_to_48k(path: str, ffmpeg: str) -> np.ndarray:
    p = subprocess.run([ffmpeg, "-v", "error", "-i", path, "-ac", "1",
                        "-ar", str(SR), "-f", "f32le", "-"],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0 or len(p.stdout) < 4:
        raise RuntimeError("音频解码失败：" + p.stderr.decode("gbk", errors="replace")[:200])
    return np.frombuffer(p.stdout, dtype="<f4").copy()


class EdgeTTSEngine:
    name = "edge-tts"

    def __init__(self, voice="zh-CN-YunxiNeural", rate="-4%", pitch="-2Hz",
                 volume="+0%", concurrency=4, ffmpeg="ffmpeg", work_dir="work",
                 cache=None):
        import edge_tts
        self.edge_tts = edge_tts
        self.voice, self.rate, self.pitch, self.volume = voice, rate, pitch, volume
        self.concurrency = max(1, int(concurrency))
        self.ffmpeg = ffmpeg
        self.tmp = os.path.join(work_dir, "tts_edge")
        os.makedirs(self.tmp, exist_ok=True)
        self.cache = cache
        self.base_pct = int(str(rate).replace("%", "").replace("+", "") or 0)

    def _rate_for(self, speed: float) -> str:
        pct = int(np.clip(self.base_pct + (speed - 1.0) * 100, -90, 90))
        return f"{pct:+d}%"

    def _k(self, text, rate):
        return cache_key("edge", self.voice, str(rate), self.pitch, self.volume, text)

    async def _one(self, i, text, rate, sem, tag="a"):
        k = self._k(text or " ", rate)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None:
                y, _ = hit
                return i, np.asarray(y, dtype=np.float32)
        # 空文本直接返回静音：pace 重排后可能产生空窗口，避免 edge-tts 写出
        # 空文件导致整批失败（空响应既不抛异常、又满足 save() 的「成功」路径）。
        if not (text or "").strip():
            y = np.zeros(SR // 20, dtype=np.float32)   # 50ms 静音
            if self.cache is not None:
                self.cache.put(k, y, SR)
            return i, y
        async with sem:
            path = os.path.join(self.tmp, f"{i:03d}_{tag}.mp3")
            last_err = "未知原因（未捕获异常，但返回音频为空）"
            for attempt in range(4):
                try:
                    # 改用 stream 收原始字节：空响应会立刻暴露，不会静默写出空文件
                    c = self.edge_tts.Communicate(text, self.voice, rate=rate,
                                                  pitch=self.pitch, volume=self.volume)
                    data = b""
                    async for msg in c.stream():
                        if msg.get("type") == "audio":
                            data += msg.get("data", b"")
                    if len(data) < 1024:
                        last_err = f"返回音频仅 {len(data)} 字节（疑似空响应 / 网络抖动）"
                        await asyncio.sleep(0.7 * (attempt + 1))
                        continue
                    with open(path, "wb") as f:
                        f.write(data)
                    y = _decode_to_48k(path, self.ffmpeg)
                    if y is None or len(y) < int(SR * 0.05):
                        last_err = f"解码后波形过短（{len(y) if y is not None else 0} 采样）"
                        await asyncio.sleep(0.7 * (attempt + 1))
                        continue
                    if self.cache is not None:
                        self.cache.put(k, y, SR)
                    return i, y
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(0.7 * (attempt + 1))
            raise RuntimeError(f"edge-tts 第 {i+1} 段合成失败：{last_err}")

    def synth_batch(self, texts, log=print):
        async def run():
            sem = asyncio.Semaphore(self.concurrency)
            tasks = [self._one(i, t or " ", self.rate, sem) for i, t in enumerate(texts)]
            res = await asyncio.gather(*tasks)
            return [y for _, y in sorted(res, key=lambda r: r[0])]
        out = asyncio.run(run())
        extra = f"  {self.cache.summary()}" if self.cache is not None else ""
        log(f"[TTS] edge-tts 合成 {len(out)} 段  voice={self.voice} rate={self.rate}{extra}")
        return [(y, SR) for y in out]

    def resynth(self, index, text, speed):
        async def run():
            sem = asyncio.Semaphore(1)
            _, y = await self._one(index, text or " ", self._rate_for(speed), sem, tag="b")
            return y
        return asyncio.run(run()), SR

    def resynth_many(self, jobs, texts, tag="r"):
        """jobs: [(index, pct), ...] —— 用绝对 rate 百分比并发重合成。

        时长与 rate 近似满足 native(pct) ≈ K / (1 + pct/100)，pipeline 用这个
        反解出每段该用的 pct，一轮就能把自然时长调到目标，不必再动波形。
        """
        async def run():
            sem = asyncio.Semaphore(self.concurrency)
            async def one(i, pct):
                return await self._one(i, texts[i] or " ",
                                       f"{int(round(pct)):+d}%", sem, tag=tag)
            return await asyncio.gather(*[one(i, p) for i, p in jobs])
        res = asyncio.run(run())          # _one 返回 (index, ndarray)
        return {i: (arr, SR) for i, arr in res}


class CosyVoiceEngine:
    """CosyVoice3 本地推理引擎（经 Gradio API 调用本机 webui 服务）。

    前置：本机已启动 CosyVoice webui（如 D:\\CosyVoiceDesktop，双击 启动webui.bat，
    端口默认 8000）。首次启动会自动从 ModelScope 下载模型。

    说明：
    - CosyVoice3 要求 prompt 文本末尾带 <|endofprompt|> 分隔标记，本类自动补，调用方无感。
    - 生成走 GPU，串行逐段合成（webui 队列并发=2，并发提交无收益）。
    - 固定 seed 保证同参数同文本结果可复现；上层 tts_cache 再做一层落盘缓存。
    - 语速：CosyVoice 的 speed 与 edge 的 rate 百分比换算为 speed = 1 + pct/100，
      限幅 [0.5, 2.0]。
    """

    name = "cosyvoice"

    def __init__(self, api="http://127.0.0.1:8000", prompt_wav="", prompt_text="",
                 mode="3s极速复刻", seed=42, speed=1.0, ffmpeg="ffmpeg",
                 work_dir="work", cache=None, timeout=600):
        from gradio_client import Client
        self.api = api.rstrip("/")
        self.prompt_wav = os.path.abspath(prompt_wav) if prompt_wav else ""
        self.prompt_text = (prompt_text or "").strip()
        self.mode = mode or "3s极速复刻"
        self.seed = int(seed)
        self.speed = float(speed)
        self.ffmpeg = ffmpeg
        self.tmp = os.path.join(work_dir, "tts_cosyvoice")
        os.makedirs(self.tmp, exist_ok=True)
        self.cache = cache
        self.timeout = int(timeout)
        self.client = Client(self.api, verbose=False)

    def _speed_for(self, pct: float) -> float:
        return float(np.clip(1.0 + float(pct) / 100.0, 0.5, 2.0))

    def _k(self, text, speed):
        return cache_key("cosyvoice", self.api, self.prompt_wav, self.prompt_text,
                         self.mode, str(self.seed), f"{speed:.3f}", text)

    def _synth_one(self, text, speed):
        """合成一段，返回 48k float32 波形。空文本返回 50ms 静音。"""
        if not (text or "").strip():
            return np.zeros(SR // 20, dtype=np.float32)
        # gradio_client>=0.10 用 file() 包装本地文件；旧版直接传路径字符串
        try:
            from gradio_client import file as gr_file
            wav_arg = gr_file(self.prompt_wav) if self.prompt_wav else None
        except ImportError:
            wav_arg = self.prompt_wav or None
        # 位置参数顺序（与 webui view_api 一致，新旧 gradio_client 均兼容）：
        # tts_text, mode, sft_dropdown, prompt_text, prompt_wav_upload,
        # prompt_wav_record, instruct_text, seed, stream, speed
        result = self.client.predict(
            text,
            self.mode,
            "",
            self.prompt_text + "<|endofprompt|>",
            wav_arg,
            None,
            "",
            self.seed,
            False,
            float(np.clip(speed, 0.5, 2.0)),
            api_name="/generate_audio")
        src = result[0] if isinstance(result, (list, tuple)) else result
        if not src or not os.path.exists(src):
            raise RuntimeError("CosyVoice 未返回音频文件")
        y = _decode_to_48k(src, self.ffmpeg)
        try:
            os.remove(src)          # gradio 临时文件，用完即删
        except OSError:
            pass
        if y is None or len(y) < int(SR * 0.05):
            raise RuntimeError("CosyVoice 返回波形过短")
        return y

    def _get(self, i, text, speed, tag="a"):
        k = self._k(text or " ", speed)
        if self.cache is not None:
            hit = self.cache.get(k)
            if hit is not None:
                return i, np.asarray(hit[0], dtype=np.float32)
        y = self._synth_one(text, speed)
        if self.cache is not None:
            self.cache.put(k, y, SR)
        return i, y

    def synth_batch(self, texts, log=print):
        out = []
        for i, t in enumerate(texts):
            _, y = self._get(i, t or " ", self.speed)
            out.append(y)
            log(f"[TTS] cosyvoice {i+1}/{len(texts)} 段完成")
        extra = f"  {self.cache.summary()}" if self.cache is not None else ""
        log(f"[TTS] cosyvoice 合成 {len(out)} 段  mode={self.mode} "
            f"seed={self.seed} speed={self.speed}{extra}")
        return [(y, SR) for y in out]

    def resynth(self, index, text, speed):
        _, y = self._get(index, text or " ", self._speed_for(speed), tag="b")
        return y, SR

    def resynth_many(self, jobs, texts, tag="r"):
        """CosyVoice 走 GPU 串行；jobs 里带各自目标 pct。"""
        res = {}
        for i, pct in jobs:
            _, y = self._get(i, texts[i] or " ", self._speed_for(pct), tag=tag)
            res[i] = (y, SR)
        return res


def make_engine(cfg, ffmpeg):
    """工厂：按 tts_engine 分支返回对应引擎。"""
    cache = None
    if cfg.get("tts_cache", True):
        root = cfg.get("tts_cache_dir") or os.path.join(cfg["work_dir"], "tts_cache")
        cache = DiskCache(root)
    engine = cfg.get("tts_engine", "edge-tts")
    if engine == "cosyvoice":
        return CosyVoiceEngine(
            api=cfg.get("cosy_api", "http://127.0.0.1:8000"),
            prompt_wav=cfg.get("cosy_prompt_wav", ""),
            prompt_text=cfg.get("cosy_prompt_text", ""),
            mode=cfg.get("cosy_mode", "3s极速复刻"),
            seed=cfg.get("cosy_seed", 42),
            speed=cfg.get("cosy_speed", 1.0),
            ffmpeg=ffmpeg, work_dir=cfg["work_dir"], cache=cache)
    return EdgeTTSEngine(
        voice=cfg["edge_voice"], rate=cfg["edge_rate"], pitch=cfg["edge_pitch"],
        concurrency=cfg.get("edge_concurrency", 4), ffmpeg=ffmpeg,
        work_dir=cfg["work_dir"], cache=cache)
