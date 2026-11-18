"""TTS 结果磁盘缓存：把「生成」变成「查表」，让成品逐样本可复现。

为什么需要它：
  * edge-tts 服务端实测确定性，但服务端模型升级 / 网络重试会产生漂移；
  * 本地 VITS 是生成式声学模型，同参数每次采样出不同波形（实测 corr≈0、
    时长差 15%），是真正的「抽卡」。
缓存按完整输入参数 + 文本做 key，第一次生成后锁死；命中后连网络都不请求。
改台词 / 换音色 / 调 rate 会自动 miss（key 变了），不会拿旧声音凑数。
"""
import os
import hashlib
import numpy as np

VERSION = "dubcache-v1"


def key(*parts):
    """所有会改变输出的参数都必须进 key，漏一个就会出现错配。"""
    s = VERSION + "||" + "||".join(str(p) for p in parts)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:24]


class DiskCache:
    def __init__(self, root):
        self.root = root
        self.hits = 0
        self.misses = 0
        if root:
            os.makedirs(root, exist_ok=True)

    def _path(self, k):
        return os.path.join(self.root, k + ".wav")

    def get(self, k):
        """返回 (float32 波形, 采样率)；未命中返回 None。"""
        if not self.root:
            self.misses += 1
            return None
        p = self._path(k)
        if not os.path.exists(p) or os.path.getsize(p) < 64:
            self.misses += 1
            return None
        try:
            import soundfile as sf
            y, sr = sf.read(p, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            self.hits += 1
            return np.ascontiguousarray(y, dtype=np.float32), int(sr)
        except Exception:
            # 半截文件 / 损坏：当未命中，下次原子覆盖
            try:
                os.remove(p)
            except OSError:
                pass
            self.misses += 1
            return None

    def put(self, k, y, sr):
        """float32 精度写盘 + 原子替换，保证重载后逐样本一致。"""
        if not self.root:
            return
        import soundfile as sf
        p = self._path(k)
        tmp = p + ".tmp"
        try:
            # .tmp 后缀无法推断容器，必须显式指定 WAV（FLOAT 保留 float32 精度）
            sf.write(tmp, np.asarray(y, dtype=np.float32), int(sr),
                     format="WAV", subtype="FLOAT")
            os.replace(tmp, p)
        finally:
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass

    def summary(self):
        return f"缓存命中 {self.hits} / 新生成 {self.misses}"

    def clear(self):
        n = 0
        if self.root and os.path.isdir(self.root):
            for f in os.listdir(self.root):
                if f.endswith((".wav", ".tmp")):
                    try:
                        os.remove(os.path.join(self.root, f))
                        n += 1
                    except OSError:
                        pass
        self.hits = self.misses = 0
        return n
