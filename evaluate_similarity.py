"""Speaker similarity (SECS) evaluation for voice cloning.

Computes cosine similarity between speaker embeddings of generated audio and the
reference speaker's real audio, using a pretrained speaker encoder (resemblyzer).
This is the key metric for voice cloning — PESQ/STOI/DNSMOS measure audio quality,
not whether the voice sounds like the target speaker.

Usage (from repo root):
    python evaluate_similarity.py                      # default dirs below
    python evaluate_similarity.py --gen-dir outputs/batch_inference --ref-dir data/test_data

Requires: pip install resemblyzer webrtcvad-wheels
"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen-dir", default=str(ROOT_DIR / "outputs" / "batch_inference"),
                        help="Directory with generated wavs (named '<stem>_generated.wav')")
    parser.add_argument("--ref-dir", default=str(ROOT_DIR / "data" / "test_data"),
                        help="Directory with real reference wavs (named '<stem>.wav')")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <gen-dir>/secs_results.json)")
    args = parser.parse_args()

    from resemblyzer import VoiceEncoder, preprocess_wav

    gen_dir, ref_dir = Path(args.gen_dir), Path(args.ref_dir)
    out_path = Path(args.out) if args.out else gen_dir / "secs_results.json"

    encoder = VoiceEncoder()

    def embed(wav_path):
        wav = preprocess_wav(str(wav_path))
        return encoder.embed_utterance(wav)

    results = {}
    for gen_file in sorted(gen_dir.glob("*_generated.wav")):
        stem = gen_file.name[: -len("_generated.wav")]
        ref_file = ref_dir / f"{stem}.wav"
        if not ref_file.exists():
            print(f"skip {stem}: no reference {ref_file}")
            continue
        sim = float(np.dot(embed(gen_file), embed(ref_file)))  # embeddings are L2-normalized
        results[stem] = sim
        print(f"{stem}: SECS = {sim:.4f}")

    if results:
        scores = list(results.values())
        summary = {
            "mean": float(np.mean(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "std": float(np.std(scores)),
            "count": len(scores),
        }
    else:
        summary = {"count": 0}

    out_path.write_text(json.dumps({"summary": summary, "per_file": results}, indent=2), encoding="utf-8")
    print(f"\nSummary: {summary}")
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
