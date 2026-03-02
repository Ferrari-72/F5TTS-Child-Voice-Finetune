"""
audio_player.py — Interactive WAV Audio Player for F5-TTS Project
==================================================================
Discovers all WAV files across the project (reference data, dialect data,
generated outputs) and lets the user browse & play them interactively.

Dependencies (pip install):
    soundfile   – WAV / FLAC decoding
    sounddevice – PortAudio playback
    numpy       – already required by TTS pipeline

Fallback:
    If the above packages are not installed the script opens every selected
    file with the Windows default media player via `os.startfile`.

Usage:
    python audio_player.py              # interactive menu
    python audio_player.py --all        # play every file sequentially
    python audio_player.py --dir <path> # browse a custom directory
    python audio_player.py --file <wav> # play a single file and exit
"""

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent.resolve()

# Ordered list of directories to scan (most-recent outputs first)
SEARCH_DIRS: dict[str, Path] = {
    "Generated – Quick Test":   ROOT_DIR / "test_outputs" / "quick_test",
    "Generated – Batch Infer":  ROOT_DIR / "test_outputs" / "batch_inference",
    "Reference – Test Data":    ROOT_DIR / "data" / "test_data",
    "Dialect – Sichuan Speech": ROOT_DIR / "data" / "dialect_data" / "speech",
    "SpeechScore – Clean":      ROOT_DIR / "speechscore" / "audios" / "clean",
    "SpeechScore – Noisy":      ROOT_DIR / "speechscore" / "audios" / "noisy",
}

DIVIDER = "─" * 65


# ──────────────────────────────────────────────────────────────────────────────
# Audio backend helpers
# ──────────────────────────────────────────────────────────────────────────────

def _try_import_audio_libs():
    try:
        import soundfile as sf
        import sounddevice as sd
        return sf, sd
    except ImportError:
        return None, None


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}" if m else f"{s:.2f}s"


def _file_meta(path: Path) -> dict:
    """Return a dict with size_kb, duration, sample_rate (best-effort)."""
    meta = {"size_kb": path.stat().st_size / 1024, "duration": None, "sample_rate": None}
    sf, _ = _try_import_audio_libs()
    if sf:
        try:
            info = sf.info(str(path))
            meta["duration"]    = info.duration
            meta["sample_rate"] = info.samplerate
        except Exception:
            pass
    return meta


# ──────────────────────────────────────────────────────────────────────────────
# File discovery
# ──────────────────────────────────────────────────────────────────────────────

def discover_audio_files(extra_dir: Path | None = None) -> list[tuple[str, Path]]:
    """
    Returns a list of (category_label, wav_path) tuples.
    Files within each category are sorted alphabetically.
    """
    dirs = dict(SEARCH_DIRS)
    if extra_dir and extra_dir.is_dir():
        dirs[f"Custom – {extra_dir.name}"] = extra_dir

    results: list[tuple[str, Path]] = []
    for label, directory in dirs.items():
        if not directory.exists():
            continue
        wav_files = sorted(directory.glob("*.wav")) + sorted(directory.glob("*.flac"))
        for f in wav_files:
            results.append((label, f))
    return results


# ──────────────────────────────────────────────────────────────────────────────
# Playback
# ──────────────────────────────────────────────────────────────────────────────

def play_file(path: Path) -> bool:
    """Play a single audio file. Returns True on success."""
    sf, sd = _try_import_audio_libs()

    if sf and sd:
        try:
            print(f"\n  ▶  {path.name}")
            audio, sr = sf.read(str(path))
            meta = _file_meta(path)
            if meta["duration"]:
                print(f"     Sample rate : {sr:,} Hz   Duration : {_format_duration(meta['duration'])}")
            sd.play(audio, sr)
            # Show a simple progress bar while playing
            total = len(audio) / sr
            start = time.time()
            try:
                while sd.get_stream().active:
                    elapsed = time.time() - start
                    pct = min(elapsed / total, 1.0)
                    bar = "█" * int(pct * 30) + "░" * (30 - int(pct * 30))
                    print(f"\r     [{bar}] {_format_duration(elapsed)} / {_format_duration(total)}",
                          end="", flush=True)
                    time.sleep(0.1)
            except Exception:
                sd.wait()
            print(f"\r     [{'█' * 30}] {_format_duration(total)} / {_format_duration(total)}  ✓")
            return True
        except Exception as exc:
            print(f"  ✗  Playback error: {exc}")
            return False
    else:
        # Fallback: open with Windows default player
        print(f"  ℹ  soundfile/sounddevice not found – opening with system player …")
        print(f"     pip install soundfile sounddevice")
        try:
            os.startfile(str(path))
            return True
        except Exception as exc:
            print(f"  ✗  Cannot open file: {exc}")
            return False


# ──────────────────────────────────────────────────────────────────────────────
# Interactive menu
# ──────────────────────────────────────────────────────────────────────────────

def _print_file_list(files: list[tuple[str, Path]]) -> None:
    current_category = None
    for idx, (category, path) in enumerate(files, start=1):
        if category != current_category:
            current_category = category
            print(f"\n  [{category}]")
        meta = _file_meta(path)
        dur_str  = _format_duration(meta["duration"]) if meta["duration"] else "  ─── "
        size_str = f"{meta['size_kb']:6.1f} KB"
        print(f"  {idx:>3}.  {path.name:<30}  {dur_str:>7}   {size_str}")


def interactive_menu(files: list[tuple[str, Path]]) -> None:
    """Present an interactive numbered menu for the user."""
    if not files:
        print("\n  ✗  No audio files found in any of the search directories.\n")
        print("     Run one of the following to generate audio first:")
        print("       python quick_inference.py")
        print("       python batch_inference.py")
        return

    while True:
        print(f"\n{'═' * 65}")
        print("  F5-TTS Audio Player")
        print(f"{'═' * 65}")
        _print_file_list(files)
        print(f"\n{DIVIDER}")
        print("  Commands:  <number>  play file     a  play all     q  quit")
        print(DIVIDER)

        raw = input("  > ").strip().lower()

        if raw in ("q", "quit", "exit"):
            print("  Bye!")
            break

        if raw in ("a", "all"):
            print(f"\n  Playing all {len(files)} files …")
            for _, path in files:
                play_file(path)
            continue

        try:
            choice = int(raw)
            if 1 <= choice <= len(files):
                play_file(files[choice - 1][1])
            else:
                print(f"  ✗  Enter a number between 1 and {len(files)}.")
        except ValueError:
            print("  ✗  Unknown command.")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio_player",
        description="Browse and play WAV/FLAC files generated by the F5-TTS pipeline.",
    )
    p.add_argument("--all",  action="store_true", help="Play every discovered file sequentially.")
    p.add_argument("--dir",  type=Path, default=None, metavar="PATH",
                   help="Additional directory to scan for audio files.")
    p.add_argument("--file", type=Path, default=None, metavar="FILE",
                   help="Play a single file and exit.")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    # Single-file mode
    if args.file:
        if not args.file.exists():
            print(f"  ✗  File not found: {args.file}")
            sys.exit(1)
        play_file(args.file)
        return

    files = discover_audio_files(extra_dir=args.dir)

    # Play-all mode (non-interactive)
    if args.all:
        if not files:
            print("  ✗  No audio files found.")
            sys.exit(1)
        print(f"  Playing {len(files)} files …")
        for _, path in files:
            play_file(path)
        return

    # Default: interactive menu
    interactive_menu(files)


if __name__ == "__main__":
    main()

