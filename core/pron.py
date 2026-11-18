"""多音字校准：合成前把念错的字换成同音字，字幕/时间轴仍保留原字。

edge-tts 只接受纯文本（不支持 SSML 注音），所以精确控制读音只能靠同音字替换。

两条途径
--------
1) 替换表 pron_fix.json：{"圈": "全", "猪圈": "猪倦"}
   · 词级最长优先：先替换 "猪圈"→"猪倦"，剩下的单个 "圈" 才走兜底规则
   · 作用在「语速均衡之后」的最终文本上，任何模式都生效
2) 行内标注：台词里写 圈[quan1] / 圈[quán] / 圈(quan1)
   · 用 pypinyin 的常用字表反查同音字替换，你只写拼音、不用自己找同音字
   · 只在该声调存在「常用同音字」时生效；像 quān（第一声）常用字只有「圈」
     自己，没有可靠同音字，会保留原字（绝不塞「悛/颧」这类生僻字或错声调）
   · 会改变字符构成，仅在「关闭语速均衡」的手动模式下可靠

fix(text)   返回用于 TTS 合成的文本
clean(text) 去掉行内标注，返回字幕 / 时间轴展示文本
"""
import os
import re
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIX_PATH = os.path.join(ROOT, "pron_fix.json")
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pinyin_map.json")

# 默认校准表：单独出现的「圈」读 quan，带「圈」的畜栏词读 juan（词级优先）
DEFAULT_FIX = {
    "_说明": "多音字校准：只改发音，不改字幕。格式 \"原词\": \"同音词\"，长词优先匹配。改完重跑即可。",
    "_示例": "猪圈->猪倦；单字圈默认读 quan（全）。行内标注写法：圈[quan1]",
    "猪圈": "猪倦",
    "圈养": "倦养",
    "圈舍": "倦舍",
    "羊圈": "羊倦",
    "牛圈": "牛倦",
    "鸡圈": "鸡倦",
    "马圈": "马倦",
    "圈": "全",
}

# 行内标注：汉字 + [拼音] / 【拼音】 / (拼音) / （拼音）；拼音可带声调符号
_MARK = re.compile(r"([\u4e00-\u9fa5])[\[【(（]([A-Za-z\u00c0-\u024f]+[1-5]?)[\]】)）]")

# 带声调韵母 -> (基本字母, 声调)
_ACCENT = {}
for _base, _tones in [("a", "\u0101\u00e1\u01ce\u00e0"),
                      ("o", "\u014d\u00f3\u01d2\u00f2"),
                      ("e", "\u0113\u00e9\u011b\u00e8"),
                      ("i", "\u012b\u00ed\u01d0\u00ec"),
                      ("u", "\u016b\u00fa\u01d4\u00f9"),
                      ("v", "\u01d6\u01d8\u01da\u01dc")]:
    for _i, _c in enumerate(_tones):
        _ACCENT[_c] = (_base, _i + 1)

_table = {}
_order = []
_pat = None
_pmap = None


# ------------------------------------------------------------------ 拼音字表
def _gb2312_chars():
    """GB2312 一级 + 二级汉字（6763 字），顺序即常用度近似。"""
    out = []
    for b1 in range(0xB0, 0xF8):
        for b2 in range(0xA1, 0xFF):
            try:
                out.append(bytes([b1, b2]).decode("gb2312"))
            except Exception:
                pass
    return out


_COMMON = None


def _get_common():
    """GB2312 一级 3755 个常用字，用于同音字优先选常用字（不选「悛」这种生僻字）。"""
    global _COMMON
    if _COMMON is None:
        s = set()
        for b1 in range(0xB0, 0xD8):
            for b2 in range(0xA1, 0xFF):
                try:
                    s.add(bytes([b1, b2]).decode("gb2312"))
                except Exception:
                    pass
        _COMMON = s
    return _COMMON


def _build_map():
    try:
        from pypinyin import pinyin, Style
    except Exception:
        return {}
    chars = _gb2312_chars()
    try:
        pys = pinyin("".join(chars), style=Style.TONE3, heteronym=False)
    except Exception:
        return {}
    m = {}
    for ch, py in zip(chars, pys):
        k = py[0] if py else ""
        if k and k[-1].isdigit():
            m.setdefault(k, []).append(ch)
    return m


def _load_map():
    if os.path.exists(MAP_PATH):
        try:
            with open(MAP_PATH, "r", encoding="utf-8") as f:
                m = json.load(f)
            if isinstance(m, dict) and m:
                return m
        except Exception:
            pass
    m = _build_map()
    if m:
        try:
            with open(MAP_PATH, "w", encoding="utf-8") as f:
                json.dump(m, f, ensure_ascii=False)
        except Exception:
            pass
    return m


def _get_pmap():
    global _pmap
    if _pmap is None:
        _pmap = _load_map()
    return _pmap


def _tone3(s):
    """quán / quan1 / quan -> quan2 / quan2 / quan"""
    s = (s or "").strip().lower()
    if not s:
        return ""
    if s[-1] in "12345":
        return s
    out, tone = "", 0
    for ch in s:
        if ch in _ACCENT:
            b, t = _ACCENT[ch]
            out += b
            tone = t
        else:
            out += ch
    out = out.replace("\u00fc", "v")
    return out + (str(tone) if tone else "")


def _lookup(pmap, key):
    if not key:
        return []
    if key in pmap:
        return pmap[key]
    if key[-1].isdigit():
        return []
    # 没写声调：合并所有声调，按常用度（GB2312 顺序）取
    out = []
    for k, lst in pmap.items():
        if k.rstrip("12345") == key:
            out.extend(lst)
    return out


def _gb_key(c):
    """GB2312 编码序 ≈ 笔画少 → 常用；用来在同音字里挑最常用的那个。"""
    try:
        return c.encode("gb2312")
    except Exception:
        return b"\xff\xff"


def _homophone(ch, py, pmap):
    """只在该声调下存在常用同音字时才替换。

    第一声 quān 这类「常用字只有原字自己」的音节，没有可靠同音字，
    硬换（悛/颧）会变成生僻字或错声调，比不换更糟 —— 此时保留原字。
    """
    key = _tone3(py)
    common = _get_common()
    hot = [c for c in _lookup(pmap, key) if c != ch and c in common]
    if hot:
        return min(hot, key=_gb_key)
    return ch


def _apply_marks(text, pmap):
    if not pmap:
        return text
    return _MARK.sub(lambda m: _homophone(m.group(1), m.group(2), pmap), text)


# ------------------------------------------------------------------ 替换表
def reload():
    """重新读 pron_fix.json（界面改完表后重新点开始就会走到这里）。"""
    global _table, _order, _pat
    tbl = {}
    if os.path.exists(FIX_PATH):
        try:
            with open(FIX_PATH, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict):
                tbl = {k: v for k, v in raw.items()
                       if not str(k).startswith("_") and isinstance(v, str) and v}
        except Exception:
            tbl = {}
    if not tbl:
        tbl = {k: v for k, v in DEFAULT_FIX.items() if not k.startswith("_")}
    _table = tbl
    _order = sorted(tbl.keys(), key=len, reverse=True)
    # 单遍扫描正则：长词优先。必须用 re.sub 一次替换，
    # 否则「一圈圈」先变「一圈一圈」，下一步单字规则又把新文本里的「圈」再换一遍。
    _pat = (re.compile("|".join(re.escape(k) for k in _order)) if _order else None)


def _ensure():
    if not _order:
        reload()


def ensure_fix_file():
    """界面按钮用：文件不存在就写一份带示例的默认表。"""
    if not os.path.exists(FIX_PATH):
        try:
            with open(FIX_PATH, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_FIX, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return FIX_PATH


# ------------------------------------------------------------------ 对外接口
def fix(text):
    """返回用于 TTS 合成的文本：先解行内标注，再套替换表（长词优先、单遍）。"""
    if not text:
        return text
    _ensure()
    if _MARK.search(text):
        text = _apply_marks(text, _get_pmap())
    if _pat is not None:
        text = _pat.sub(lambda m: _table[m.group(0)], text)
    return text


def clean(text):
    """去掉行内读音标注，返回字幕 / 时间轴展示的原字文本。"""
    return _MARK.sub(lambda m: m.group(1), text or "")
