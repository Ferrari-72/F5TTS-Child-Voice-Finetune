# 🎙️ F5-TTS Child Voice Fine-Tuning
### AIAA2205 Assignment 2 — Text-to-Speech Fine-Tuning

> Fine-tune [F5-TTS](https://github.com/SWivid/F5-TTS) to clone a **child voice** and synthesize high-quality Mandarin Chinese speech.  
> 基于 F5-TTS 的**儿童音色克隆**与中文语音合成项目。

---

## 📋 Project Overview / 项目概述

| Item | Detail |
|---|---|
| Base Model | F5-TTS v1 Base (pretrained) |
| Fine-tune Dataset | `child-tts` — 220 short Mandarin child speech recordings (~14 min) |
| Language | Mandarin Chinese (普通话) |
| Hardware | NVIDIA RTX 4060 Laptop (8 GB VRAM) |
| Reference Audio | `data/child-tts/000018.wav` — "城里有好多游乐场，可好玩儿了！" |

**Pipeline:**  
`Data Preprocessing` → `Download Pretrained Model` → `Fine-tuning` → `Inference` → `Objective Evaluation`

---

## 🚀 Quick Start / 快速开始

### 1. Install Dependencies / 安装依赖

```bash
pip install -r requirements.txt
python install_deps.py        # install evaluation packages
conda install -c conda-forge ffmpeg -y   # required for audio loading
```

### 2. Generate Speech (GUI Player) / 生成语音并播放

```bash
# Generate 5 sample audio files
python generate_samples.py

# Open the interactive GUI player (click to play)
python audio_player_gui.py
```

### 3. Generate Custom Speech / 自定义文本合成

```bash
python speech_synthesis.py "今天天气很好，我们去公园玩吧！"
```

### 4. Data Preprocessing / 数据预处理

> ⚠️ Run only once — outputs are cached in `data/child-tts_processed/`

```bash
python src/f5_tts/train/datasets/data_prepare.py \
    ./data/child-tts ./data/child-tts_processed
```

### 5. Fine-tuning / 模型微调

Open and run `tts_finetuning_main.ipynb` in Jupyter / Colab.

> **⚠️ Important:** Use the pretrained model's vocab (`ckpts/child-tts/F5TTS_v1_Base/vocab.txt`, 2545 lines / ~9327 chars) for both training and inference. Do NOT replace it with the auto-generated dataset vocab.

### 6. Batch Inference / 批量推理

```bash
python batch_inference.py
```

### 7. Quality Evaluation / 音频质量评估

```bash
python evaluate_quality.py
```

---

## 📁 Project Structure / 项目结构

```
.
├── 📓 tts_finetuning_main.ipynb        Main fine-tuning notebook
├── 📓 tts_finetuning_simplified.ipynb  Simplified version
│
├── 🐍 audio_player_gui.py              GUI player — click to play any audio
├── 🐍 audio_player.py                  CLI audio player
├── 🐍 generate_samples.py              Generate sample WAV files
├── 🐍 speech_synthesis.py              Single-text TTS inference
├── 🐍 batch_inference.py               Batch inference on test set
├── 🐍 evaluate_quality.py              Multi-metric audio evaluation
├── 🐍 hparam_search.py                 Hyperparameter search
├── 🐍 quick_inference.py               Quick single-file inference test
├── 🐍 install_deps.py                  Install evaluation dependencies
├── 🐍 inspect_notebook.py              Notebook structure inspector
│
├── src/f5_tts/                         F5-TTS core source (submodule)
│   ├── configs/                        Model configs (.yaml)
│   ├── infer/                          Inference utilities & CLI
│   ├── Models/                         DiT / UNetT / CFM model code
│   └── train/                          Training scripts & data prep
│
├── data/
│   ├── child-tts/                      Raw training WAVs (220 files)
│   ├── child-tts_processed/            Preprocessed Arrow dataset
│   └── test_data/                      Evaluation WAVs (LF / YZ series)
│
├── ckpts/
│   ├── child-tts/
│   │   ├── pretrained_model_1250000.safetensors   ← pretrained base
│   │   └── F5TTS_v1_Base/vocab.txt                ← correct vocab (2545 tokens)
│   └── child-tts_processed/
│       └── model_last.pt              ← fine-tuned checkpoint
│
├── test_outputs/
│   ├── quick_test/                     Single-run generated WAVs
│   └── batch_inference/                Batch-run generated WAVs
│
├── speechscore/                        Speech quality evaluation toolkit
├── docs/                               Extended documentation
└── 图表/                               Analysis plots & charts
```

---

## 🎧 Audio Player / 音频播放器

`audio_player_gui.py` — a dark-theme Tkinter GUI that automatically discovers all WAV files in the project.

| Action | How |
|---|---|
| Play | Double-click a file **or** select + click ▶ Play |
| Stop | Click ■ Stop |
| Open folder | Click 📂 Open |
| Refresh file list | Click 🔄 Refresh |

```bash
python audio_player_gui.py
```

---

## 📊 Evaluation Metrics / 评估指标

| Metric | Range | Meaning |
|---|---|---|
| PESQ | 1.0 – 4.5 | Perceptual speech quality |
| STOI | 0.0 – 1.0 | Short-time objective intelligibility |
| CSIG | 1.0 – 5.0 | Signal distortion |
| CBAK | 1.0 – 5.0 | Background noise |
| COVL | 1.0 – 5.0 | Overall quality |
| DNSMOS | — | DNS challenge MOS score |
| SRMR | — | Speech-to-reverberation modulation ratio |

---

## ⚙️ Key Inference Parameters / 推理参数

| Parameter | Recommended | Description |
|---|---|---|
| `nfe_step` | 32 – 50 | Diffusion denoising steps (higher = better quality, slower) |
| `cfg_strength` | 2.0 | Classifier-free guidance (higher = more faithful to reference) |
| `ref_audio` | `000018.wav` | Voice-cloning reference audio |
| `vocab_file` | `ckpts/child-tts/F5TTS_v1_Base/vocab.txt` | **Must** use pretrained vocab |

---

## 🐛 Known Issues / 已知问题

### Fine-tuned model generates noise
The current `model_last.pt` outputs near-silence noise instead of speech.  
**Root cause:** The fine-tuning run used the auto-generated dataset vocab (464 tokens) instead of the pretrained model vocab (2545 tokens), causing the text embedding to diverge.  
**Workaround:** Use `pretrained_model_1250000.safetensors` for inference (works correctly).  
**Fix:** Re-run fine-tuning with `vocab_file = ckpts/child-tts/F5TTS_v1_Base/vocab.txt`.

---

## 📚 Documentation / 文档

See `docs/` for detailed write-ups:

| File | Content |
|---|---|
| `SCORE_OPTIMIZATION_METHODS.md` | How to maximize evaluation scores |
| `OPTIMIZATION_SUMMARY.md` | Parameter optimization results |
| `BATCH_EVALUATION.md` | Batch evaluation guide |
| `PROJECT_SUMMARY.md` | Full project summary |
| `IMPROVEMENTS_SUMMARY.md` | Code improvement log |

---

## 🙏 Acknowledgements / 致谢

- [F5-TTS](https://github.com/SWivid/F5-TTS) — base model & architecture
- [SpeechScore](https://github.com/auspicious3000/SpeechScore) — evaluation toolkit

---

## 📝 License

See [LICENSE](LICENSE).
