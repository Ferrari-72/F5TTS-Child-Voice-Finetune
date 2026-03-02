"""
generate_samples.py — Generate sample audio files with the fine-tuned F5-TTS model
====================================================================================
Generates several short Chinese speech samples and saves them to
  test_outputs/quick_test/
so they appear automatically in the GUI player (audio_player_gui.py).

Usage:
    python generate_samples.py
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

ROOT_DIR = Path(__file__).parent.resolve()
os.chdir(ROOT_DIR)

# ── Model / reference audio ──────────────────────────────────────────────────
MODEL_CFG   = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"

# ⚠️  微调模型 (model_last.pt) 训练时 vocab 配置有误，checkpoint 输出纯噪声。
#     暂用预训练模型做语音克隆，效果正常。
#     如需修复微调：用正确的 9331-char vocab 重新训练，见 README。
CKPT_FILE   = ROOT_DIR / "ckpts" / "child-tts" / "pretrained_model_1250000.safetensors"
VOCAB_FILE  = ROOT_DIR / "ckpts" / "child-tts" / "F5TTS_v1_Base" / "vocab.txt"  # 9327字符，预训练 vocab

REF_AUDIO   = ROOT_DIR / "data"  / "child-tts" / "000018.wav"
REF_TEXT    = "城里有好多游乐场，可好玩儿了！"  # 000018.wav 的真实转录文本

INFER_CLI   = ROOT_DIR / "src" / "f5_tts" / "infer" / "infer_cli.py"
OUTPUT_DIR  = ROOT_DIR / "test_outputs" / "quick_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Inference parameters ─────────────────────────────────────────────────────
NFE_STEP     = 32
CFG_STRENGTH = 2.0

# ── Sample texts to generate ─────────────────────────────────────────────────
SAMPLES = [
    ("sample_01_weather.wav",   "今天天气真好，小朋友们出去玩吧。"),
    ("sample_02_school.wav",    "同学们，上课了，请把书本拿出来。"),
    ("sample_03_story.wav",     "从前有一只小白兔，它住在森林里的小木屋里。"),
    ("sample_04_greeting.wav",  "你好啊，好久不见，最近过得怎么样？"),
    ("sample_05_park.wav",      "城里有好多游乐场，可好玩儿了！"),
]

# ─────────────────────────────────────────────────────────────────────────────

def normalize_audio(path: Path, target_peak: float = 0.85) -> None:
    """Peak-normalize the WAV file so it's clearly audible."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(path))
        peak = np.max(np.abs(audio))
        if peak > 0 and peak < target_peak * 0.5:   # only boost if significantly quiet
            audio = audio * (target_peak / peak)
            audio = np.clip(audio, -1.0, 1.0)
            sf.write(str(path), audio, sr)
            print(f"       volume boosted {target_peak/peak:.1f}x → peak={target_peak:.2f}")
    except Exception as e:
        print(f"       normalize warning: {e}")


def run_inference(gen_text: str, output_file: str) -> bool:
    out_path = OUTPUT_DIR / output_file
    cmd = [
        sys.executable, str(INFER_CLI),
        "--model_cfg",   str(MODEL_CFG),
        "--ckpt_file",   str(CKPT_FILE),
        "--vocab_file",  str(VOCAB_FILE),
        "--ref_audio",   str(REF_AUDIO),
        "--ref_text",    REF_TEXT,
        "--gen_text",    gen_text,
        "--output_dir",  str(OUTPUT_DIR),
        "--output_file", output_file,
        "--nfe_step",     str(NFE_STEP),
        "--cfg_strength", str(CFG_STRENGTH),
        "--target_rms",   "0.2",   # 提高输出音量（默认约0.016，此处提至0.2）
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    try:
        result = subprocess.run(
            cmd, env=env, cwd=str(ROOT_DIR),
            capture_output=True, text=True, encoding="utf-8", timeout=300
        )
        if result.returncode == 0 and out_path.exists() and out_path.stat().st_size > 0:
            normalize_audio(out_path)          # 保证音量可听
            kb = out_path.stat().st_size / 1024
            print(f"  OK  {output_file}  ({kb:.0f} KB)")
            return True
        else:
            err = (result.stderr or result.stdout)[:300].strip()
            print(f"  FAIL  {output_file}")
            if err:
                print(f"       {err}")
            return False
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT  {output_file}")
        return False
    except Exception as exc:
        print(f"  ERROR  {output_file}: {exc}")
        return False


def main():
    print("=" * 60)
    print("F5-TTS  Sample Generator")
    print(f"Model : {CKPT_FILE.name}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Texts : {len(SAMPLES)}")
    print("=" * 60)

    ok = 0
    for filename, text in SAMPLES:
        print(f"\n[{SAMPLES.index((filename, text))+1}/{len(SAMPLES)}]  {text}")
        if run_inference(text, filename):
            ok += 1

    print("\n" + "=" * 60)
    print(f"Done — {ok}/{len(SAMPLES)} files generated in:")
    print(f"  {OUTPUT_DIR}")
    if ok:
        print("\nOpen audio_player_gui.py to listen!")
    print("=" * 60)


if __name__ == "__main__":
    main()

