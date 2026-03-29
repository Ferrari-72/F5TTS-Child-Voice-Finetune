"""
audio_player_gui.py — GUI Audio Player for F5-TTS Project
==========================================================
双击或点击 Play 即可播放，无需命令行。
Double-click any file or press Play — no command line needed.

依赖 (pip install):
    soundfile   sounddevice   numpy
    (均已在 TTS 环境中安装)
"""

import os
import time
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ── 目录配置 ────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.resolve()

SEARCH_DIRS: dict[str, Path] = {
    "Generated · Quick Test": ROOT_DIR / "test_outputs" / "quick_test",
    "Generated · Batch Infer": ROOT_DIR / "test_outputs" / "batch_inference",
    "Generated · Speech": ROOT_DIR / "outputs" / "generated_speech",
    "Generated · Outputs": ROOT_DIR / "outputs" / "quick_test",
    "Reference · Test Data": ROOT_DIR / "data" / "test_data",
    "Dialect · Sichuan Speech": ROOT_DIR / "data" / "dialect_data" / "speech",
    "SpeechScore · Clean": ROOT_DIR / "speechscore" / "audios" / "clean",
    "SpeechScore · Noisy": ROOT_DIR / "speechscore" / "audios" / "noisy",
}

# ── 音频后端 ─────────────────────────────────────────────────────────────────
try:
    import soundfile as sf
    import sounddevice as sd

    AUDIO_BACKEND = "sounddevice"
except ImportError:
    AUDIO_BACKEND = "system"


def _get_info(path: Path):
    """返回 (duration_sec, sample_rate) 或 (None, None)"""
    if AUDIO_BACKEND == "sounddevice":
        try:
            info = sf.info(str(path))
            return info.duration, info.samplerate
        except Exception:
            pass
    return None, None


def _fmt_dur(sec):
    if sec is None:
        return "─"
    m, s = divmod(int(sec), 60)
    return f"{m:02d}:{s:02d}"


# ── 文件发现 ─────────────────────────────────────────────────────────────────
def discover() -> list[tuple[str, Path, float | None, int | None]]:
    rows = []
    for label, directory in SEARCH_DIRS.items():
        if not directory.exists():
            continue
        for ext in ("*.wav", "*.flac"):
            for p in sorted(directory.glob(ext)):
                dur, sr = _get_info(p)
                rows.append((label, p, dur, sr))
    return rows


# ── 播放线程 ─────────────────────────────────────────────────────────────────
_play_thread: threading.Thread | None = None
_stop_flag = threading.Event()


def _play_worker(path: Path, progress_cb, done_cb):
    _stop_flag.clear()
    if AUDIO_BACKEND == "sounddevice":
        try:
            audio, sr = sf.read(str(path))
            total = len(audio) / sr
            chunk = int(sr * 0.05)  # 50 ms chunks for responsive stop
            pos = 0
            stream = sd.OutputStream(
                samplerate=sr, channels=1 if audio.ndim == 1 else audio.shape[1], dtype=audio.dtype
            )
            stream.start()
            while pos < len(audio) and not _stop_flag.is_set():
                block = audio[pos : pos + chunk]
                stream.write(block)
                pos += chunk
                progress_cb(pos / len(audio), pos / sr, total)
            stream.stop()
            stream.close()
        except Exception as e:
            done_cb(error=str(e))
            return
    else:
        # Fallback: Windows default player (non-blocking)
        os.startfile(str(path))
    done_cb(error=None)


# ── 主界面 ───────────────────────────────────────────────────────────────────
class AudioPlayerGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("F5-TTS  Audio Player")
        self.geometry("820x540")
        self.configure(bg="#1e1e2e")
        self.resizable(True, True)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._files: list[tuple[str, Path, float | None, int | None]] = []
        self._current_path: Path | None = None
        self._playing = False

        self._build_ui()
        self._load_files()

    # ── 构建界面 ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Treeview 颜色
        style.configure(
            "Treeview",
            background="#2a2a3e",
            foreground="#cdd6f4",
            rowheight=26,
            fieldbackground="#2a2a3e",
            font=("Consolas", 10),
        )
        style.configure("Treeview.Heading", background="#313244", foreground="#89b4fa", font=("Segoe UI", 10, "bold"))
        style.map("Treeview", background=[("selected", "#45475a")])

        # ── 顶部标题 ──
        header = tk.Frame(self, bg="#181825", pady=8)
        header.pack(fill="x")
        tk.Label(
            header, text="🎵  F5-TTS  Audio Player", bg="#181825", fg="#cba6f7", font=("Segoe UI", 14, "bold")
        ).pack()

        # ── 文件列表 ──
        frame_tree = tk.Frame(self, bg="#1e1e2e")
        frame_tree.pack(fill="both", expand=True, padx=12, pady=(8, 0))

        cols = ("category", "filename", "duration", "sr", "size")
        self.tree = ttk.Treeview(frame_tree, columns=cols, show="headings", selectmode="browse")
        self.tree.heading("category", text="Category")
        self.tree.heading("filename", text="File Name")
        self.tree.heading("duration", text="Duration")
        self.tree.heading("sr", text="Sample Rate")
        self.tree.heading("size", text="Size")
        self.tree.column("category", width=200, anchor="w")
        self.tree.column("filename", width=240, anchor="w")
        self.tree.column("duration", width=80, anchor="center")
        self.tree.column("sr", width=100, anchor="center")
        self.tree.column("size", width=80, anchor="center")

        vsb = ttk.Scrollbar(frame_tree, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # 双击 / 回车 播放
        self.tree.bind("<Double-1>", lambda e: self._play_selected())
        self.tree.bind("<Return>", lambda e: self._play_selected())

        # ── 信息栏 ──
        self.info_var = tk.StringVar(value="选择一个文件，然后点击 Play 或双击播放")
        info_bar = tk.Label(
            self, textvariable=self.info_var, bg="#181825", fg="#a6e3a1", font=("Segoe UI", 9), anchor="w", padx=10
        )
        info_bar.pack(fill="x", pady=(4, 0))

        # ── 进度条 ──
        self.progress = ttk.Progressbar(self, orient="horizontal", mode="determinate", maximum=100)
        self.progress.pack(fill="x", padx=12, pady=4)

        self.time_var = tk.StringVar(value="00:00 / 00:00")
        tk.Label(self, textvariable=self.time_var, bg="#1e1e2e", fg="#bac2de", font=("Consolas", 9)).pack()

        # ── 控制按钮 ──
        btn_frame = tk.Frame(self, bg="#1e1e2e", pady=8)
        btn_frame.pack()

        btn_cfg = dict(width=10, font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", bd=0, padx=8, pady=5)

        self.btn_play = tk.Button(
            btn_frame, text="▶  Play", bg="#89b4fa", fg="#1e1e2e", command=self._play_selected, **btn_cfg
        )
        self.btn_play.grid(row=0, column=0, padx=6)

        self.btn_stop = tk.Button(
            btn_frame, text="■  Stop", bg="#f38ba8", fg="#1e1e2e", command=self._stop, state="disabled", **btn_cfg
        )
        self.btn_stop.grid(row=0, column=1, padx=6)

        self.btn_open = tk.Button(
            btn_frame, text="📂  Open", bg="#a6e3a1", fg="#1e1e2e", command=self._open_in_explorer, **btn_cfg
        )
        self.btn_open.grid(row=0, column=2, padx=6)

        tk.Button(btn_frame, text="🔄  Refresh", bg="#fab387", fg="#1e1e2e", command=self._load_files, **btn_cfg).grid(
            row=0, column=3, padx=6
        )

        # ── 状态栏 ──
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(
            self, textvariable=self.status_var, bg="#181825", fg="#6c7086", font=("Segoe UI", 8), anchor="w", padx=10
        ).pack(fill="x", side="bottom")

    # ── 加载文件 ──────────────────────────────────────────────────────────────
    def _load_files(self):
        self.status_var.set("Scanning …")
        self.tree.delete(*self.tree.get_children())
        self._files = discover()

        for i, (cat, path, dur, sr) in enumerate(self._files):
            size_kb = f"{path.stat().st_size / 1024:.0f} KB"
            sr_str = f"{sr:,} Hz" if sr else "─"
            tag = "even" if i % 2 == 0 else "odd"
            self.tree.insert(
                "", "end", iid=str(i), tags=(tag,), values=(cat, path.name, _fmt_dur(dur), sr_str, size_kb)
            )

        self.tree.tag_configure("odd", background="#2a2a3e")
        self.tree.tag_configure("even", background="#252535")

        count = len(self._files)
        self.status_var.set(f"Found {count} audio file{'s' if count != 1 else ''}")
        if count == 0:
            self.info_var.set("⚠  No audio files found. Run quick_inference.py or batch_inference.py first.")

    # ── 播放逻辑 ──────────────────────────────────────────────────────────────
    def _play_selected(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "请先点击选择一个音频文件")
            return

        idx = int(sel[0])
        _, path, dur, sr = self._files[idx]

        if self._playing:
            self._stop()
            time.sleep(0.15)

        self._current_path = path
        self._playing = True
        self.btn_play.config(state="disabled")
        self.btn_stop.config(state="normal")
        self.progress["value"] = 0
        total_str = _fmt_dur(dur)
        self.info_var.set(f"▶  {path.name}   ({sr:,} Hz   {total_str})" if sr else f"▶  {path.name}")
        self.status_var.set(f"Playing: {path.name}")

        global _play_thread
        _play_thread = threading.Thread(
            target=_play_worker,
            args=(path, self._on_progress, self._on_done),
            daemon=True,
        )
        _play_thread.start()

    def _on_progress(self, fraction: float, elapsed: float, total: float):
        """Called from worker thread — schedule UI update on main thread."""
        self.after(0, self._update_progress, fraction, elapsed, total)

    def _update_progress(self, fraction, elapsed, total):
        self.progress["value"] = fraction * 100
        self.time_var.set(f"{_fmt_dur(elapsed)} / {_fmt_dur(total)}")

    def _on_done(self, error=None):
        self.after(0, self._playback_finished, error)

    def _playback_finished(self, error):
        self._playing = False
        self.btn_play.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.progress["value"] = 0
        if error:
            self.status_var.set(f"Error: {error}")
            messagebox.showerror("Playback Error", error)
        else:
            self.status_var.set("Finished")
            self.time_var.set("00:00 / 00:00")

    def _stop(self):
        _stop_flag.set()
        self._playing = False
        self.btn_play.config(state="normal")
        self.btn_stop.config(state="disabled")
        self.status_var.set("Stopped")

    # ── 在资源管理器中打开 ────────────────────────────────────────────────────
    def _open_in_explorer(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showinfo("No Selection", "请先点击选择一个文件")
            return
        idx = int(sel[0])
        path = self._files[idx][1]
        os.startfile(str(path.parent))

    def _on_close(self):
        _stop_flag.set()
        self.destroy()


# ── 入口 ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if AUDIO_BACKEND == "system":
        print("[WARNING] soundfile / sounddevice not found.")
        print("          pip install soundfile sounddevice")
        print("          Falling back to Windows default player.\n")
    app = AudioPlayerGUI()
    app.mainloop()
