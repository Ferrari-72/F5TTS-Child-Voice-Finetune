"""Generate demo comparison set: real vs pretrained (zero-shot) vs fine-tuned.

For each wav in data/test_data (or a subset), transcribe with Whisper, then synthesize
the same text with both the pretrained base model and the fine-tuned checkpoint.
Outputs land in demo/ as <stem>_real.wav / <stem>_pretrained.wav / <stem>_finetuned.wav,
plus demo/transcripts.json.

Usage (from repo root):
    python make_demo.py [--n 5] [--nfe-step 50]
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import soundfile as sf
from omegaconf import OmegaConf

from src.f5_tts.Models import DiT
from src.f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder

TEST_DIR = ROOT_DIR / "data" / "test_data"
DEMO_DIR = ROOT_DIR / "demo"
MODEL_CFG = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"
VOCAB_FILE = ROOT_DIR / "ckpts" / "child-tts" / "vocab.txt"
PRETRAINED = ROOT_DIR / "ckpts" / "child-tts" / "pretrained_model_1250000.safetensors"
FINETUNED = ROOT_DIR / "ckpts" / "child-tts_pinyin" / "model_last.pt"
REF_AUDIO = ROOT_DIR / "data" / "child-tts" / "000018.wav"
REF_TEXT = "城里有好多游乐场，可好玩儿了！"


def build_model(ckpt, vocoder, model_cfg):
    return load_model(DiT, model_cfg.model.arch, str(ckpt), mel_spec_type="vocos", vocab_file=str(VOCAB_FILE))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="number of test files to use")
    parser.add_argument("--nfe-step", type=int, default=50)
    parser.add_argument("--cfg-strength", type=float, default=2.0)
    args = parser.parse_args()

    DEMO_DIR.mkdir(exist_ok=True)
    test_files = sorted(TEST_DIR.glob("*.wav"))[: args.n]
    if not test_files:
        raise SystemExit(f"no wavs in {TEST_DIR}")

    transcripts_path = DEMO_DIR / "transcripts.json"
    if transcripts_path.exists():
        transcripts = json.loads(transcripts_path.read_text(encoding="utf-8"))
        print(f"Reusing {transcripts_path}")
        for f in test_files:
            shutil.copy(f, DEMO_DIR / f"{f.stem}_real.wav")
    else:
        import whisper

        print("Loading Whisper (small)...")
        asr = whisper.load_model("small")
        transcripts = {}
        for f in test_files:
            text = asr.transcribe(str(f), language="zh")["text"].strip()
            transcripts[f.stem] = text
            print(f"{f.stem}: {text}")
            shutil.copy(f, DEMO_DIR / f"{f.stem}_real.wav")

    print("Loading vocoder...")
    vocoder = load_vocoder(vocoder_name="vocos", is_local=False, local_path="")
    model_cfg = OmegaConf.load(str(MODEL_CFG))

    for tag, ckpt in [("pretrained", PRETRAINED), ("finetuned", FINETUNED)]:
        print(f"\n=== {tag}: {ckpt.name} ===")
        model = build_model(ckpt, vocoder, model_cfg)
        for f in test_files:
            out = DEMO_DIR / f"{f.stem}_{tag}.wav"
            audio, sr, _ = infer_process(
                str(REF_AUDIO), REF_TEXT, transcripts[f.stem],
                model, vocoder, mel_spec_type="vocos",
                nfe_step=args.nfe_step, cfg_strength=args.cfg_strength,
            )
            sf.write(str(out), audio, sr)
            print(f"  wrote {out.name}")
        del model

    (DEMO_DIR / "transcripts.json").write_text(
        json.dumps(transcripts, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nDone. {len(test_files)} samples x (real + pretrained + finetuned) -> demo/")


if __name__ == "__main__":
    main()
