"""Re-tokenize an existing processed dataset to pinyin without re-running Whisper.

The original data_prepare.py stored raw Chinese transcripts in raw.arrow, which is
incompatible with the pretrained F5-TTS pinyin vocab (every Chinese char -> OOV -> 0).
This script reuses the existing transcripts and only redoes the text tokenization.

Usage (from repo root):
    python src/f5_tts/train/datasets/retokenize_to_pinyin.py \
        --src data/child-tts_processed --dst data/child-tts_pinyin --wav-dir data/child-tts
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[4]))  # repo root

from datasets.arrow_writer import ArrowWriter
from datasets import Dataset as Dataset_
from tqdm import tqdm

from src.f5_tts.Models.utils import convert_char_to_pinyin


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/child-tts_processed", help="Existing processed dataset dir")
    parser.add_argument("--dst", default="data/child-tts_pinyin", help="Output dataset dir")
    parser.add_argument("--wav-dir", default="data/child-tts",
                        help="Directory of the wav files; audio_path is rewritten to '<wav-dir>/<filename>'")
    args = parser.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    dst.mkdir(exist_ok=True, parents=True)

    table = Dataset_.from_file(str(src / "raw.arrow"))
    rows = table.to_list()
    print(f"Loaded {len(rows)} rows from {src / 'raw.arrow'}")

    texts = [r["text"] for r in rows]
    converted = convert_char_to_pinyin(texts, polyphone=True)

    with ArrowWriter(path=str(dst / "raw.arrow"), writer_batch_size=50) as writer:
        for row, pinyin_tokens in tqdm(zip(rows, converted), total=len(rows), desc="Writing pinyin dataset"):
            writer.write({
                "audio_path": (Path(args.wav_dir) / Path(row["audio_path"]).name).as_posix(),
                "text": pinyin_tokens,
                "duration": row["duration"],
            })

    with open(dst / "duration.json", "w", encoding="utf-8") as f:
        json.dump({"duration": [r["duration"] for r in rows]}, f, ensure_ascii=False)

    # informational vocab of pinyin tokens actually seen
    vocab = sorted({tok for tokens in converted for tok in tokens})
    with open(dst / "vocab.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(vocab) + "\n")

    print(f"Done. {len(rows)} samples, {len(vocab)} distinct pinyin tokens -> {dst}")
    print(f"Sample: {texts[0]} -> {converted[0]}")


if __name__ == "__main__":
    main()
