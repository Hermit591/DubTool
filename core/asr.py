"""第 3 步：把每个句子组转成文字。三种来源，软件不依赖任何外部智能体。

  sensevoice : 本地 ONNX 离线识别（默认，模型在 models/sensevoice/）
  manual     : 用户直接粘贴台词（每行一句）
  srt        : 导入已有字幕，用字幕时间轴覆盖波形分句
"""
import os
import re
import numpy as np


def clean(text: str) -> str:
    """SenseVoice 会带 <|zh|><|NEUTRAL|><|Speech|> 之类的标签，清掉只留正文。"""
    t = re.sub(r"<\|[^|]*\|>", "", text or "")
    t = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", t)
    return t.strip().strip("，, 。.")


class SenseVoiceASR:
    def __init__(self, model_dir: str, num_threads: int = 4):
        import sherpa_onnx
        model = os.path.join(model_dir, "model.int8.onnx")
        tokens = os.path.join(model_dir, "tokens.txt")
        missing = [p for p in (model, tokens) if not os.path.exists(p)]
        if missing:
            raise FileNotFoundError(
                "本地识别模型缺失：" + ", ".join(missing) +
                "\n请运行「一键安装依赖.bat」下载模型，或把 ASR 引擎改成「手动输入台词」。")
        self.rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model, tokens=tokens, num_threads=num_threads,
            language="zh", use_itn=True, debug=False)

    def transcribe(self, y: np.ndarray, sr: int = 16000) -> str:
        if len(y) < sr * 0.10:
            return ""
        s = self.rec.create_stream()
        s.accept_waveform(sr, np.asarray(y, dtype=np.float32))
        self.rec.decode_stream(s)
        return clean(s.result.text)


def transcribe_groups(y, sr, chunks, groups, engine, cfg, log=print):
    """返回每个句子组的文字。"""
    n = len(groups)
    if engine == "manual":
        raw = cfg.get("manual_text", "")
        lines = [clean(x) for x in raw.splitlines() if x.strip()]
        if not lines:
            raise ValueError("手动模式需要在台词框里粘贴台词（每行一句）")
        if len(lines) == n:
            return lines
        # 行数与句数不一致：按句数比例合并/拆分，不静默丢内容
        out = []
        for i in range(n):
            a, b = round(i * len(lines) / n), round((i + 1) * len(lines) / n)
            out.append("".join(lines[max(a, 0):max(b, a + 1)]))
        log(f"[ASR] 手动台词 {len(lines)} 行 → 对齐到 {n} 个句子组")
        return out

    if engine == "srt":
        path = cfg.get("srt_path", "")
        items = parse_srt(path)
        if not items:
            raise ValueError("SRT 里没解析到字幕")
        out = []
        for g in groups:
            a, b = chunks[g[0]][0], chunks[g[-1]][1]
            # 取与这个窗口重叠最多的那条字幕
            best, score = "", 0.0
            for sa, sb, st in items:
                ov = max(0.0, min(b, sb) - max(a, sa))
                if ov > score:
                    best, score = st, ov
            out.append(clean(best))
        return out

    asr = SenseVoiceASR(cfg["sensevoice_dir"])
    out = []
    for gi, idxs in enumerate(groups, 1):
        a = chunks[idxs[0]][0]
        b = chunks[idxs[-1]][1]
        seg = y[int(a * sr):int(b * sr)]
        txt = asr.transcribe(seg, sr)
        out.append(txt)
        log(f"[ASR] 句 {gi}/{len(groups)}  {a:.2f}-{b:.2f}s  {txt}")
    return out


def parse_srt(path: str) -> list:
    """解析 SRT → [(start_sec, end_sec, text), ...]"""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        body = f.read()
    def to_sec(s):
        h, m, rest = s.split(":")
        sec, ms = rest.replace(",", ".").split(".")
        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms.ljust(3, "0")[:3]) / 1000
    out = []
    for block in re.split(r"\n\s*\n", body):
        m = re.search(r"(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})", block)
        if not m:
            continue
        text = " ".join(x.strip() for x in block.splitlines()[2:] if x.strip())
        if text:
            out.append((to_sec(m.group(1)), to_sec(m.group(2)), text))
    return out
