# F5-TTS Child Voice Fine-Tuning

**[🎧 Listen to the demo](https://ferrari-72.github.io/F5TTS-Child-Voice-Finetune/)** — real vs. zero-shot vs. fine-tuned, inline players.

Fine-tuning [F5-TTS](https://github.com/SWivid/F5-TTS) v1 Base to clone a **Mandarin child's voice**
from only **~14 minutes of speech** (224 short clips), on a single **RTX 4060 Laptop (8 GB VRAM)**.

中文说明见各 `docs/*.md`（本文以英文为主）。

| Item | Detail |
|---|---|
| Base model | F5-TTS v1 Base (DiT, 335M params) |
| Dataset | `child-tts` — 224 Mandarin child-speech clips, ~14 min, single speaker |
| Hardware | RTX 4060 Laptop, 8 GB VRAM (dev/smoke test) · Kaggle P100 16 GB (full run) |
| Reference clip | `data/child-tts/000018.wav` — "城里有好多游乐场，可好玩儿了！" |
| Metrics | PESQ / STOI / DNSMOS / SRMR (quality) + **SECS** (speaker similarity) |

## Analysis: why this is hard, and where the work went

**The setup is adversarial by construction.** A 335M-parameter diffusion transformer, ~14 minutes
of single-speaker data, a budget of *one* real training run — and a training pipeline that had
been silently broken in four places, so the naive "just run fine-tune" produces pure noise with
no error message. Three numbers frame the whole project: **335M params / 14 min / 2500 steps**.

### Failure → evidence → root cause (the actual debugging loop)

Each bug was isolated by a falsifiable check, not by reading code top-to-bottom:

| Symptom | Diagnostic check | Root cause | Fix |
|---|---|---|---|
| Output = white noise | Train 100 steps, listen: still noise → not undertraining | `load_checkpoint()` gutted to `return 0`; model trained from random init | Restore official loader (safetensors → EMA + online) |
| Loss falls but output stays noise | Print vocab size at runtime: 464 ≠ 2545 | Dataset char vocab used with pretrained pinyin embeddings | Force pretrained `vocab.txt` everywhere |
| Text conditioning has no effect | Decode token ids back: all zeros | Chinese chars never converted to pinyin → every char OOV (id 0) | Retokenize Arrow dataset to pinyin (`retokenize_to_pinyin.py`) |
| "Fixed" checkpoint still garbage | Trace which vocab inference loads | `VOCAB_FIX.md` pointed inference at the wrong vocab | Delete the doc, pin correct vocab in scripts |

Validation after the fix: a 100-step smoke test synthesized speech that **Whisper ASR transcribes
back as exactly the target sentence** — a cheap, decisive end-to-end check before spending the
one training run.

### Design decisions worth noticing

- **Warmup rescaled to the run, not copied from the paper.** Official default warmup (20k steps)
  exceeds the entire run (2.5k steps) — LR would never leave warmup. Warmup set to 10% of total steps.
- **Checkpoint selection by SECS, not by loss.** Flow-matching loss does not track perceptual
  speaker similarity; every intermediate checkpoint was kept (`keep_last_n_checkpoints=-1`),
  batch-synthesized, and scored with a speaker encoder afterwards.
- **Honest metrics.** SECS parity with the zero-shot baseline (0.526 vs 0.526) is reported as-is,
  with the noise caveats — see [Evaluation](#evaluation).
- **One-shot discipline.** Smoke test → full run on free cloud GPU (Kaggle P100) → post-hoc
  selection, so the single real run never gets wasted on a misconfiguration.

### Workload at a glance

| Area | What was actually done |
|---|---|
| Training core | Repaired `Trainer.load_checkpoint()`, added `--mixed_precision`, slim periodic saves (model+EMA only), resume priority fix |
| Data | Built pinyin retokenizer that reuses existing transcripts (no Whisper re-run); rewrote audio paths |
| Infra | Automated Kaggle pipeline (dataset upload, kernel script, 5 debug iterations on mount paths/CLI deprecations/missing deps); GitHub Pages demo site |
| Evaluation | SECS harness (resemblyzer), quality-metrics integration (SpeechScore), ASR round-trip validation, checkpoint-selection sweep |
| Product | Gradio UI (`app.py`), CLI inference, batch inference with resume, demo comparison set |

## The debugging story

Full postmortem of the four stacked bugs summarized in the table above, with the forensic detail:
[`docs/DEBUG_NOTES.md`](docs/DEBUG_NOTES.md) (中文). The short version: pretrained weights were
never loaded, the vocab was mismatched, pinyin conversion was skipped at data prep, and a "fix"
doc inverted the last one — each independently fatal, all silent.

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

> The fine-tuned demo comes from the **full 50-epoch run** (2500 steps, trained on a Kaggle P100),
> checkpoint selected by SECS across intermediate saves — see the Evaluation section below.

## Evaluation

Speaker similarity (SECS, resemblyzer cosine vs. the real recording), measured on the 5 demo pairs.
Checkpoint selection across the full 50-epoch run (2500 steps, Kaggle P100, fp16):

| Model | SECS (mean) |
|---|---|
| Pretrained (zero-shot) | 0.526 |
| Fine-tuned @ 1250 steps | 0.520 |
| **Fine-tuned @ 2500 steps (released)** | **0.526** |
| Fine-tuned @ last (EMA variant) | 0.514 |

Honest reading: with ~14 minutes of data and 2500 steps, the fine-tuned model reaches parity with
the zero-shot baseline on this 5-sample SECS probe (differences are within noise, std ≈ 0.03).
The perceptible win is in **accent/prosody matching the child speaker** — listen to the demo pairs —
not in the embedding-cosine number. Treat the result as "small-data fine-tuning preserves speaker
similarity while adapting style", not a SECS improvement claim.

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
