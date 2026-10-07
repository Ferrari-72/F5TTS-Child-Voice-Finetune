# F5-TTS Child Voice Fine-Tuning

Fine-tuning [F5-TTS](https://github.com/SWivid/F5-TTS) v1 Base to clone a **Mandarin child's voice**
from only **~14 minutes of speech** (224 short clips), on a single **RTX 4060 Laptop (8 GB VRAM)**.

中文说明见各 `docs/*.md`（本文以英文为主）。

| Item | Detail |
|---|---|
| Base model | F5-TTS v1 Base (DiT, 335M params) |
| Dataset | `child-tts` — 224 Mandarin child-speech clips, ~14 min, single speaker |
| Hardware | RTX 4060 Laptop, 8 GB VRAM |
| Reference clip | `data/child-tts/000018.wav` — "城里有好多游乐场，可好玩儿了！" |
| Metrics | PESQ / STOI / DNSMOS / SRMR (quality) + **SECS** (speaker similarity) |

## The debugging story

The first fine-tuned checkpoint produced pure noise. Root-cause analysis found **four stacked
bugs**, each independently fatal (full write-up: [`docs/DEBUG_NOTES.md`](docs/DEBUG_NOTES.md)):

1. **Pretrained weights were never loaded** — `Trainer.load_checkpoint()` had been gutted to
   `return 0`, so "fine-tuning" was actually training from random init on 14 minutes of data.
2. **Vocab mismatch** — training used the auto-generated 464-char dataset vocab instead of the
   pretrained 2545-token pinyin vocab, breaking the text embedding shape.
3. **Missing pinyin conversion at data prep** — raw Chinese transcripts were stored in the Arrow
   dataset; with the pinyin vocab every Chinese character maps to OOV (id 0), i.e. the model saw
   empty text conditioning. The official pipeline converts to pinyin *before* writing the dataset.
4. **An inverted "fix" doc** — `VOCAB_FIX.md` pointed inference scripts at the 464-char vocab,
   which made even a correct checkpoint produce garbage (pinyin tokens are all OOV in a
   Chinese-char vocab).

After the fixes, a **1-epoch smoke test (100 steps)** already produces fully intelligible speech:
Whisper ASR on the synthesized clip transcribes it back as exactly the target sentence
(`今天天气很好，我们去公园玩吧！`). See `demo/smoke_test_100steps.wav`.

## Quick start

```bash
conda activate f5-tts          # Python 3.10+, torch + CUDA
pip install -r requirements.txt
```

The full pipeline lives in **`tts_finetuning_main.ipynb`** (5 steps). CLI summary:

```bash
# 1. Data: pinyin-retokenize existing transcripts (no Whisper re-run)
python src/f5_tts/train/datasets/retokenize_to_pinyin.py \
    --src data/child-tts_processed --dst data/child-tts_pinyin --wav-dir data/child-tts

# 2. Smoke test (~5 min) — verify vocab size, checkpoint loading, falling loss
python -X utf8 src/f5_tts/train/finetune_cli.py \
    --epochs 1 --batch_size_per_gpu 2 --save_per_updates 30 --last_per_updates 60

# 3. Full training (the one real run)
python -X utf8 src/f5_tts/train/finetune_cli.py \
    --epochs 50 --batch_size_per_gpu 4 --learning_rate 1e-5 --scheduler cosine \
    --save_per_updates 500 --last_per_updates 100 --keep_last_n_checkpoints -1

# 4. Checkpoint selection: batch-infer + score each candidate
python batch_inference.py         # synthesize held-out test set
python evaluate_quality.py        # PESQ / STOI / DNSMOS / SRMR ...
python evaluate_similarity.py     # SECS speaker similarity (pick by this first)

# 5. Inference
python speech_synthesis.py "今天天气很好，我们去公园玩吧！"   # CLI
python app.py                                                 # Gradio web UI
```

If you only get one training run, follow [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md):
smoke test first, keep all intermediate checkpoints, select by SECS + DNSMOS afterwards.

## Key settings

| Setting | Value | Note |
|---|---|---|
| tokenizer | `custom` + pretrained `vocab.txt` (2545 pinyin tokens) | never the dataset char vocab |
| lr / schedule | 1e-5, cosine, auto warmup = 10% of total steps | official default 20k warmup > total steps |
| batch | 4 samples (8 GB VRAM) | drop to 2 if OOM |
| `nfe_step` / `cfg_strength` | 32–50 / 2.0 (quality runs: 150 / 3.0) | diffusion sampling steps / guidance |

## Demo

Real held-out recordings vs. zero-shot pretrained vs. fine-tuned (same text, same reference clip).
Click a file on GitHub to listen.

| Sample | Text | Real | Pretrained (zero-shot) | Fine-tuned |
|---|---|---|---|---|
| LF-0002 | 你叫什么名字？ | [▶](demo/LF-0002_real.wav) | [▶](demo/LF-0002_pretrained.wav) | [▶](demo/LF-0002_finetuned.wav) |
| LF-0004 | 你吃饭了吗？ | [▶](demo/LF-0004_real.wav) | [▶](demo/LF-0004_pretrained.wav) | [▶](demo/LF-0004_finetuned.wav) |
| YZ-0002 | 你叫什么名字？ | [▶](demo/YZ-0002_real.wav) | [▶](demo/YZ-0002_pretrained.wav) | [▶](demo/YZ-0002_finetuned.wav) |
| YZ-0003 | 好久不见最近怎么样 | [▶](demo/YZ-0003_real.wav) | [▶](demo/YZ-0003_pretrained.wav) | [▶](demo/YZ-0003_finetuned.wav) |
| YZ-0004 | 你吃饭了吗？ | [▶](demo/YZ-0004_real.wav) | [▶](demo/YZ-0004_pretrained.wav) | [▶](demo/YZ-0004_finetuned.wav) |

> The current fine-tuned demo comes from a **200-step validation run** (the smoke-test checkpoint)
> whose purpose is proving the fixed pipeline end-to-end. Full training is a single command —
> see [`docs/TRAINING_RUNBOOK.md`](docs/TRAINING_RUNBOOK.md).

## Evaluation

Speaker similarity (SECS, resemblyzer cosine vs. the real recording), measured on the 5 demo pairs:

| Model | SECS (mean) |
|---|---|
| Pretrained (zero-shot) | 0.526 |
| Fine-tuned (200-step validation ckpt) | 0.526 |

Honest reading: at 200 warmup-phase steps the fine-tuned model has not yet moved past the
(zero-shot) baseline on speaker similarity — the numbers validate the *pipeline*, not the final
quality. Full-run results (PESQ / STOI / DNSMOS / SECS across intermediate checkpoints) get
filled in after the complete training run.

## Project structure

```
├── tts_finetuning_main.ipynb   # the whole pipeline, 5 steps
├── app.py                      # Gradio web demo
├── speech_synthesis.py         # single-text CLI inference
├── batch_inference.py          # batch inference on the test set (resumable)
├── evaluate_quality.py         # PESQ/STOI/DNSMOS/SRMR ...
├── evaluate_similarity.py      # SECS speaker similarity
├── hparam_search.py            # nfe_step / cfg_strength search
├── src/f5_tts/                 # F5-TTS source (training fixes live here)
├── docs/
│   ├── TRAINING_RUNBOOK.md     # one-shot training checklist (中文)
│   └── DEBUG_NOTES.md          # the four-bug postmortem (中文)
├── data/                       # wavs + Arrow datasets (gitignored)
├── ckpts/                      # pretrained + fine-tuned checkpoints (gitignored)
└── demo/                       # sample outputs
```

## Acknowledgements

- [F5-TTS](https://github.com/SWivid/F5-TTS) — base model & architecture
- [SpeechScore](https://github.com/auspicious3000/SpeechScore) — quality metrics
- [resemblyzer](https://github.com/resemble-ai/Resemblyzer) — speaker encoder for SECS

## License

[MIT](LICENSE)
