# -*- coding: utf-8 -*-
"""视频配音工具 · 波形时间轴零漂移

双击「启动.bat」即可运行；本程序不依赖 WorkBuddy 或任何智能体。
流程：抽音轨 → 波形切发声窗口 → 识别/导入台词 → 选音色逐段合成 →
      按采样点压回原窗口 → 封装视频。
"""
import os
import sys
import json
import queue
import warnings
import threading
import subprocess
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

warnings.filterwarnings("ignore", message=".*pkg_resources.*")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import config, voices, pipeline, pron

ROOT = os.path.dirname(os.path.abspath(__file__))
Q = queue.Queue()
CANCEL = threading.Event()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("视频配音工具 · 时间轴零漂移")
        sw = self.winfo_screenheight()
        h = min(900, max(640, (sw or 900) - 80))
        self.geometry(f"1000x{h}")
        self.minsize(880, 600)
        self.cfg = config.load()
        self.voice_map = {}
        self.result = None
        self.worker = None
        self._build()
        self._load_cfg()
        self.after(100, self._pump)

    # ---------------- 界面 ----------------
    def _build(self):
        style = ttk.Style()
        try:
            style.theme_use("vista")
        except Exception:
            pass
        f = ("Microsoft YaHei UI", 10)
        fb = ("Microsoft YaHei UI", 10, "bold")
        # 可滚动容器：界面控件较多，窗口高度不够时整体滚动，
        # 确保最底部的「运行」按钮和日志始终可达（不再被顶出可视区）。
        outer = ttk.Frame(self)
        outer.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(outer, borderwidth=0, highlightthickness=0)
        vbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        vbar.pack(side="right", fill="y")
        main = ttk.Frame(self.canvas, padding=10)
        self.canvas.create_window((0, 0), window=main, anchor="nw")
        main.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        head = ttk.Frame(main)
        head.pack(fill="x", pady=(0, 8))
        ttk.Label(head, text="视频配音工具", font=("Microsoft YaHei UI", 15, "bold")).pack(side="left")
        ttk.Label(head, text="  先按波形切时间轴 → 再合成语音压回原窗口，逐采样点对齐",
                  foreground="#666").pack(side="left")

        # 1 文件
        g1 = ttk.LabelFrame(main, text=" 1 · 文件 ", padding=8)
        g1.pack(fill="x", pady=4)
        self.in_var = tk.StringVar()
        self.out_var = tk.StringVar()
        self.tl_var = tk.StringVar()
        self._row(g1, 0, "输入视频/音频", self.in_var, self._pick_in, f)
        self._row(g1, 1, "输出成品", self.out_var, self._pick_out, f)
        r = ttk.Frame(g1); r.grid(row=2, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(r, text="复用时间轴", font=f, width=14).pack(side="left")
        ttk.Entry(r, textvariable=self.tl_var, font=f).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="选择…", command=self._pick_tl).pack(side="left")
        ttk.Button(r, text="清除", command=lambda: self.tl_var.set("")).pack(side="left", padx=4)
        ttk.Label(g1, text="（选了时间轴就跳过识别。手动微操：①关掉第4步『语速均衡』→ JSON 里的 text/end/voice_end 全部直接生效；"
                  "②想保留自动均衡、只锁住你手改的某段→给那段加 \"locked\": true 字段）",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).grid(row=3, column=1, sticky="w")

        # 2 音色
        g2 = ttk.LabelFrame(main, text=" 2 · 声音 ", padding=8)
        g2.pack(fill="x", pady=4)
        r = ttk.Frame(g2); r.pack(fill="x")
        ttk.Label(r, text="音色", font=f, width=14).pack(side="left")
        self.voice_var = tk.StringVar()
        self.voice_cb = ttk.Combobox(r, textvariable=self.voice_var, font=f, width=62, state="readonly")
        self.voice_cb.pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="试听", command=self._preview).pack(side="left")
        self.voice_info = ttk.Label(g2, text="", foreground="#2a6fb0", font=f)
        self.voice_info.pack(anchor="w", pady=(4, 2))
        r2 = ttk.Frame(g2); r2.pack(fill="x")
        ttk.Label(r2, text="语速", font=f, width=14).pack(side="left")
        self.rate_var = tk.IntVar(value=-4)
        ttk.Scale(r2, from_=-30, to=30, variable=self.rate_var, orient="horizontal",
                  command=self._show_rate).pack(side="left", fill="x", expand=True, padx=4)
        self.rate_lb = ttk.Label(r2, text="-4%", font=f, width=8); self.rate_lb.pack(side="left")
        ttk.Label(r2, text="  音调", font=f).pack(side="left")
        self.pitch_var = tk.IntVar(value=-2)
        ttk.Scale(r2, from_=-12, to=12, variable=self.pitch_var, orient="horizontal",
                  command=self._show_pitch).pack(side="left", fill="x", expand=True, padx=4)
        self.pitch_lb = ttk.Label(r2, text="-2Hz", font=f, width=8); self.pitch_lb.pack(side="left")
        self.voice_cb.bind("<<ComboboxSelected>>", self._voice_changed)

        # 3 台词
        g3 = ttk.LabelFrame(main, text=" 3 · 台词来源 ", padding=8)
        g3.pack(fill="both", pady=4)
        r = ttk.Frame(g3); r.pack(fill="x")
        ttk.Label(r, text="来源", font=f, width=14).pack(side="left")
        self.asr_var = tk.StringVar(value="manual")
        cb = ttk.Combobox(r, textvariable=self.asr_var, font=f, width=24, state="readonly",
                          values=["manual", "srt", "sensevoice"])
        cb.pack(side="left", padx=4)
        ttk.Label(r, text="  手动每行一句 / 导入 SRT / SenseVoice 离线识别(需另下模型)",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).pack(side="left")
        self.srt_var = tk.StringVar()
        r1 = ttk.Frame(g3); r1.pack(fill="x", pady=2)
        ttk.Label(r1, text="SRT 文件", font=f, width=14).pack(side="left")
        ttk.Entry(r1, textvariable=self.srt_var, font=f).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r1, text="选择…", command=self._pick_srt).pack(side="left")
        self.text = scrolledtext.ScrolledText(g3, height=6, font=f, wrap="word")
        self.text.pack(fill="both", expand=True, pady=(4, 0))
        ttk.Label(g3, text="识别结果会回填到这里；「manual」模式下直接在这里编辑台词（每行一句）",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).pack(anchor="w")
        r2 = ttk.Frame(g3); r2.pack(fill="x", pady=(4, 0))
        ttk.Label(r2, text="多音字读错？", font=f).pack(side="left")
        ttk.Button(r2, text="打开读音校准表", command=self._open_pron).pack(side="left", padx=4)
        ttk.Label(r2, text="pron_fix.json 写 \"圈\": \"全\"（只改发音、字幕不变）；某读音无常用同音字时自动保留原字",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=6)

        # 4 参数
        g4 = ttk.LabelFrame(main, text=" 4 · 波形切分参数 ", padding=8)
        g4.pack(fill="x", pady=4)
        self.gap_var = tk.DoubleVar(value=0.60)
        self.sil_var = tk.DoubleVar(value=0.20)
        self.db_var = tk.DoubleVar(value=-38)
        self.conc_var = tk.IntVar(value=4)
        for i, (lb, var, w) in enumerate([("句子间隔(s)", self.gap_var, 7),
                                          ("最短静音(s)", self.sil_var, 7),
                                          ("静音阈值(dB)", self.db_var, 7),
                                          ("合成并发", self.conc_var, 7)]):
            ttk.Label(g4, text=lb, font=f).grid(row=0, column=i * 2, padx=(0, 4))
            ttk.Entry(g4, textvariable=var, width=w, font=f).grid(row=0, column=i * 2 + 1, padx=(0, 14))
        self.pace_var = tk.BooleanVar(value=True)
        self.borrow_var = tk.DoubleVar(value=0.90)
        self.pmid_var = tk.DoubleVar(value=0.18)
        self.pbnd_var = tk.DoubleVar(value=0.55)
        r4b = ttk.Frame(g4); r4b.grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))
        ttk.Checkbutton(r4b, text="语速均衡（消除忽快忽慢，建议开启）",
                        variable=self.pace_var).pack(side="left")
        ttk.Label(r4b, text="   可借静音(s)", font=f).pack(side="left")
        ttk.Entry(r4b, textvariable=self.borrow_var, width=6, font=f).pack(side="left", padx=4)
        ttk.Label(r4b, text="句内停顿(s)", font=f).pack(side="left", padx=(10, 0))
        ttk.Entry(r4b, textvariable=self.pmid_var, width=5, font=f).pack(side="left", padx=2)
        ttk.Label(r4b, text="句尾停顿(s)", font=f).pack(side="left", padx=(8, 0))
        ttk.Entry(r4b, textvariable=self.pbnd_var, width=5, font=f).pack(side="left", padx=2)
        ttk.Label(r4b, text="停顿越大越舒缓；越小越紧凑",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=8)
        self.tone_var = tk.BooleanVar(value=True)
        self.tone_str_var = tk.DoubleVar(value=0.85)
        self.stretch_var = tk.BooleanVar(value=False)
        r4c = ttk.Frame(g4); r4c.grid(row=2, column=0, columnspan=8, sticky="w", pady=(6, 0))
        ttk.Checkbutton(r4c, text="音色一致（消除忽清忽糊，建议开启）",
                        variable=self.tone_var).pack(side="left")
        ttk.Label(r4c, text="   强度", font=f).pack(side="left")
        ttk.Entry(r4c, textvariable=self.tone_str_var, width=6, font=f).pack(side="left", padx=4)
        ttk.Checkbutton(r4c, text="允许波形变速（旧行为，会发蒙）",
                        variable=self.stretch_var).pack(side="left", padx=12)
        ttk.Label(r4c, text="0~1；统一频谱形状与响度，不再对波形做相位拉伸",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=6)
        self.cache_var = tk.BooleanVar(value=True)
        r4d = ttk.Frame(g4); r4d.grid(row=3, column=0, columnspan=8, sticky="w", pady=(6, 0))
        ttk.Checkbutton(r4d, text="锁定合成结果（缓存，重跑逐样本一致）",
                        variable=self.cache_var).pack(side="left")
        ttk.Button(r4d, text="清空缓存", command=self._clear_cache).pack(side="left", padx=10)
        ttk.Label(r4d, text="edge-tts 云端确定性合成；缓存命中时断网也能原样复现",
                  foreground="#888", font=("Microsoft YaHei UI", 9)).pack(side="left", padx=4)

        # 5 运行
        g5 = ttk.LabelFrame(main, text=" 5 · 运行 ", padding=8)
        g5.pack(fill="both", expand=True, pady=4)
        r = ttk.Frame(g5); r.pack(fill="x")
        self.run_btn = ttk.Button(r, text="开始配音", command=self._start)
        self.run_btn.pack(side="left")
        self.cancel_btn = ttk.Button(r, text="取消", command=self._cancel, state="disabled")
        self.cancel_btn.pack(side="left", padx=6)
        ttk.Button(r, text="打开输出目录", command=self._open_out).pack(side="left", padx=6)
        ttk.Button(r, text="查看说明", command=self._open_readme).pack(side="left")
        self.status = ttk.Label(r, text="就绪", foreground="#2a6fb0", font=fb)
        self.status.pack(side="right")
        self.pb = ttk.Progressbar(g5, mode="determinate", maximum=100)
        self.pb.pack(fill="x", pady=6)
        self.log = scrolledtext.ScrolledText(g5, height=9, font=("Consolas", 9), wrap="word",
                                              background="#fafafa")
        self.log.pack(fill="both", expand=True)

    def _row(self, parent, row, label, var, cmd, font):
        r = ttk.Frame(parent); r.grid(row=row, column=0, columnspan=3, sticky="ew", pady=2)
        ttk.Label(r, text=label, font=font, width=14).pack(side="left")
        ttk.Entry(r, textvariable=var, font=font).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="浏览…", command=cmd).pack(side="left")

    def _on_mousewheel(self, event):
        # 在日志 / 输入框 / 下拉框内部滚动时交给控件自己处理，不联动画布
        if isinstance(event.widget, (tk.Text, tk.Entry, ttk.Combobox, tk.Listbox)):
            return
        try:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    def _clear_cache(self):
        from core.tts_cache import DiskCache
        root = os.path.join(self.cfg["work_dir"], "tts_cache")
        n = DiskCache(root).clear()
        self._log(f"已清空合成缓存（{n} 个文件）。下次运行会重新向 TTS 引擎请求。")

    # ---------------- 配置 ----------------
    def _load_cfg(self):
        c = self.cfg
        self.in_var.set(c.get("input", ""))
        self.out_var.set(c.get("output", ""))
        self.tl_var.set(c.get("timeline_file", ""))
        self.srt_var.set(c.get("srt_path", ""))
        self.asr_var.set(c.get("asr_engine", "manual"))
        self.gap_var.set(c.get("speaker_gap", 0.60))
        self.sil_var.set(c.get("min_silence", 0.20))
        self.db_var.set(c.get("noise_db", -38))
        self.conc_var.set(c.get("edge_concurrency", 4))
        self.pace_var.set(bool(c.get("pace_balance", True)))
        self.borrow_var.set(float(c.get("pace_borrow", 0.90)))
        self.pmid_var.set(float(c.get("pace_pause_mid", 0.18)))
        self.pbnd_var.set(float(c.get("pace_pause_bnd", 0.55)))
        self.tone_var.set(bool(c.get("tone_match", True)))
        self.tone_str_var.set(float(c.get("tone_strength", 0.85)))
        self.stretch_var.set(bool(c.get("allow_stretch", False)))
        self.cache_var.set(bool(c.get("tts_cache", True)))
        try:
            self.rate_var.set(int(str(c.get("edge_rate", "-4%")).replace("%", "")))
            self.pitch_var.set(int(str(c.get("edge_pitch", "-2Hz")).replace("Hz", "")))
        except Exception:
            pass
        self._show_rate(); self._show_pitch()
        self._reload_voices(c.get("voice") or voices.default_id())
        self._log(f"就绪。音色库：{len(voices.library())} 个（在线神经语音）")
        self._log("提示：请先在「3 · 台词来源」填入台词或导入 SRT；合成需联网。")

    def _reload_voices(self, selected=None):
        vals, self.voice_map = [], {}
        for group, items in voices.grouped().items():
            for it in items:
                show = f"【{group}】{it['label']}"
                vals.append(show)
                self.voice_map[show] = it
        self.voice_cb["values"] = vals
        target = None
        for show, it in self.voice_map.items():
            if it["id"] == selected:
                target = show
                break
        if target is None and vals:
            target = vals[0]
        if target:
            self.voice_var.set(target)
            self._voice_changed()

    def _collect(self):
        c = dict(self.cfg)
        c.update({
            "input": self.in_var.get().strip(),
            "output": self.out_var.get().strip(),
            "timeline_file": self.tl_var.get().strip(),
            "srt_path": self.srt_var.get().strip(),
            "asr_engine": self.asr_var.get(),
            "manual_text": self.text.get("1.0", "end").strip(),
            "speaker_gap": float(self.gap_var.get()),
            "min_silence": float(self.sil_var.get()),
            "noise_db": float(self.db_var.get()),
            "edge_concurrency": int(self.conc_var.get()),
            "pace_balance": bool(self.pace_var.get()),
            "pace_borrow": float(self.borrow_var.get()),
            "pace_pause_mid": float(self.pmid_var.get()),
            "pace_pause_bnd": float(self.pbnd_var.get()),
            "tone_match": bool(self.tone_var.get()),
            "tone_strength": float(self.tone_str_var.get()),
            "allow_stretch": bool(self.stretch_var.get()),
            "tts_cache": bool(self.cache_var.get()),
            "edge_rate": f"{int(self.rate_var.get()):+d}%",
            "edge_pitch": f"{int(self.pitch_var.get()):+d}Hz",
            "voice": self.voice_map.get(self.voice_var.get(), {"id": voices.default_id()})["id"],
        })
        return voices.apply(c, c["voice"])

    # ---------------- 事件 ----------------
    def _pick_in(self):
        p = filedialog.askopenfilename(title="选择视频或音频",
                                       filetypes=[("视频/音频", "*.mp4 *.mkv *.mov *.avi *.mp3 *.wav *.m4a *.flv"),
                                                  ("所有文件", "*.*")])
        if not p:
            return
        self.in_var.set(p)
        # 每次选新文件都按输入名自动生成输出：<原名>_配音.mp4（音频则 .wav）
        stem = os.path.splitext(os.path.basename(p))[0]
        ext = os.path.splitext(p)[1].lower()
        suffix = ".wav" if ext in (".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg") else ".mp4"
        self.out_var.set(os.path.join(self.cfg["out_dir"], stem + "_配音" + suffix))

    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="成品保存位置", defaultextension=".mp4",
                                         filetypes=[("MP4 视频", "*.mp4")])
        if p:
            self.out_var.set(p)

    def _pick_tl(self):
        p = filedialog.askopenfilename(title="选择时间轴 JSON", filetypes=[("JSON", "*.json")])
        if p:
            self.tl_var.set(p)

    def _pick_srt(self):
        p = filedialog.askopenfilename(title="选择 SRT 字幕", filetypes=[("SRT", "*.srt")])
        if p:
            self.srt_var.set(p)
            self.asr_var.set("srt")
            with open(p, "r", encoding="utf-8-sig", errors="replace") as f:
                self.text.delete("1.0", "end")
                self.text.insert("1.0", f.read())

    def _voice_changed(self, *_):
        it = self.voice_map.get(self.voice_var.get())
        if not it:
            return
        # 音色下拉就是引擎开关：把引擎名直接打出来，避免改配置以为换了引擎。
        tip = ("在线神经语音（需联网）" if it["engine"] == "edge-tts"
               else "本地大模型（需先启动 CosyVoice 服务，参数见 config.json 的 cosy_* 项）")
        color = "#2a6fb0" if it["engine"] == "edge-tts" else "#0f6e56"
        self.voice_info.config(
            text=f"引擎 {it['engine']}  |  {it['label']}  |  {tip}  |  ID {it['id']}",
            foreground=color)
        self._show_rate(); self._show_pitch()

    def _show_rate(self, *_):
        self.rate_lb.config(text=f"{int(self.rate_var.get()):+d}%")

    def _show_pitch(self, *_):
        self.pitch_lb.config(text=f"{int(self.pitch_var.get()):+d}Hz")

    def _open_out(self):
        p = os.path.dirname(self.out_var.get().strip()) or self.cfg["out_dir"]
        os.makedirs(p, exist_ok=True)
        try:
            os.startfile(p)
        except Exception:
            self._log("打开目录失败：" + p)

    def _open_readme(self):
        p = os.path.join(ROOT, "README.md")
        if os.path.exists(p):
            try:
                os.startfile(p)
            except Exception:
                self._log("打开说明失败")

    def _open_pron(self):
        try:
            path = pron.ensure_fix_file()
            os.startfile(path)
            self._log("读音校准表：" + path)
            self._log("改完保存后重新点「开始配音」即生效（只改发音，字幕仍显示原字）。")
        except Exception as e:
            self._log("打开读音校准表失败：%s" % e)

    # ---------------- 后台任务 ----------------
    def _log(self, msg):
        Q.put(("log", str(msg)))

    def _pump(self):
        try:
            while True:
                kind, payload = Q.get_nowait()
                if kind == "log":
                    self.log.insert("end", payload + "\n")
                    self.log.see("end")
                elif kind == "progress":
                    pct, msg = payload
                    self.pb["value"] = pct
                    self.status.config(text=f"{int(pct)}% · {msg[:28]}")
                elif kind == "done":
                    self._finished(payload)
                elif kind == "error":
                    self._failed(payload)
                elif kind == "text":
                    self.text.delete("1.0", "end")
                    self.text.insert("1.0", payload)
                elif kind == "voices":
                    self._reload_voices(payload)
        except queue.Empty:
            pass
        self.after(100, self._pump)

    def _busy(self, on):
        self.run_btn.config(state="disabled" if on else "normal")
        self.cancel_btn.config(state="normal" if on else "disabled")

    def _start(self):
        c = self._collect()
        if not c["input"] or not os.path.exists(c["input"]):
            messagebox.showwarning("缺少输入", "请先选择要配音的视频/音频文件")
            return
        config.save(c)
        self.cfg = c
        CANCEL.clear()
        self._busy(True)
        self.log.delete("1.0", "end")
        self.pb["value"] = 0
        self._log(f"开始：{c['input']}")
        self._log(f"音色：{self.voice_var.get()}")
        self.worker = threading.Thread(target=self._run_worker, args=(c,), daemon=True)
        self.worker.start()

    def _run_worker(self, c):
        try:
            def log(m): Q.put(("log", m))
            def prog(p, m): Q.put(("progress", (p, m)))
            r = pipeline.run(c, log=log, progress=prog, cancel=CANCEL.is_set)
            Q.put(("text", "\n".join(r.get("group_texts", []))))
            Q.put(("done", r))
        except pipeline.Cancelled:
            Q.put(("log", "已取消"))
            Q.put(("done", {"cancelled": True}))
        except Exception as e:
            Q.put(("log", traceback.format_exc()))
            Q.put(("error", f"{type(e).__name__}: {e}"))

    def _cancel(self):
        CANCEL.set()
        self.status.config(text="正在取消…")

    def _finished(self, r):
        self._busy(False)
        if r.get("cancelled"):
            self.status.config(text="已取消")
            return
        v = r.get("verify", {})
        self._log(f"校验：视频 {v.get('duration_video')}s / 音轨 {v.get('duration_track')}s / "
                  f"漂移 {v.get('drift_ms')}ms，{v.get('segments')} 个窗口，异常 {v.get('bad_segments')}")
        self.status.config(text="完成")
        self.result = r
        messagebox.showinfo("完成", f"成品：\n{r.get('output')}\n\n"
                                   f"漂移 {v.get('drift_ms')}ms，异常窗口 {v.get('bad_segments')} 个")

    def _failed(self, msg):
        self._busy(False)
        self.status.config(text="失败")
        messagebox.showerror("出错了", msg[:600])

    # ---------------- 附加功能 ----------------
    def _preview(self):
        it = self.voice_map.get(self.voice_var.get())
        if not it:
            return
        c = self._collect()
        self._log("正在生成试听…")
        t = threading.Thread(target=self._preview_worker, args=(c, it), daemon=True)
        t.start()

    def _preview_worker(self, c, it):
        try:
            from core import tts
            from core.config import resolve_ffmpeg
            ff = resolve_ffmpeg(c)
            text = "大家好，这是一段试听语音，用来确认当前音色是否合适。"
            e = tts.make_engine(dict(c, edge_concurrency=1), ff)
            (y, sr) = e.synth_batch([text], log=lambda m: None)[0]
            import soundfile as sf
            path = os.path.join(c["work_dir"], "试听.wav")
            sf.write(path, y / max(float(abs(y).max()), 1e-6) * 0.9, sr)
            os.startfile(path)
            Q.put(("log", "试听已播放：" + path))
        except Exception as e:
            Q.put(("log", f"试听失败：{type(e).__name__}: {e}"))


if __name__ == "__main__":
    App().mainloop()
