# 训练执行手册（TRAINING RUNBOOK）

> 只有一次完整训练机会时，按本手册执行。核心思路：**用 5 分钟冒烟测试把所有致命错误
> 提前暴露，再跑正式训练，最后从多个中间 checkpoint 里选最优**。

## 0. 前置检查

```bash
conda activate f5-tts
cd <repo root>

# 数据：224 条 wav + 拼音化数据集
ls data/child-tts | wc -l          # 应为 225（含可能的 metadata）
ls data/child-tts_pinyin/          # raw.arrow / duration.json / vocab.txt

# 预训练权重和 vocab
ls ckpts/child-tts/                # pretrained_model_1250000.safetensors + vocab.txt
```

如果还没有 `data/child-tts_pinyin`（text 必须是拼音 token 列表，不是中文字符串）：

```bash
python src/f5_tts/train/datasets/retokenize_to_pinyin.py \
    --src data/child-tts_processed --dst data/child-tts_pinyin --wav-dir data/child-tts
```

验证 token 覆盖率（缺失应 ≤ 个位数）：

```bash
python -X utf8 -c "
pretrained = set(open('ckpts/child-tts/vocab.txt', encoding='utf-8').read().splitlines())
used = set(open('data/child-tts_pinyin/vocab.txt', encoding='utf-8').read().splitlines())
print('missing:', sorted(used - pretrained))"
```

## 1. 冒烟测试（约 5 分钟，不算正式训练）

```bash
python -X utf8 src/f5_tts/train/finetune_cli.py \
    --epochs 1 --batch_size_per_gpu 2 --save_per_updates 30 --last_per_updates 60
```

必须逐项确认的四个检查点：

| 检查项 | 期望日志 | 如果不符 |
|---|---|---|
| vocab 大小 | `vocab : 2545` | vocab 文件指错了，检查 `--tokenizer_path` |
| 权重加载 | `Loading checkpoint: pretrained_model_1250000.safetensors` | 没加载=从头训练，输出必为噪声 |
| LR 调度 | `Auto warmup: N steps`（N ≈ 总步数 10%） | warmup 过大则 LR 全程接近 0 |
| loss | 逐步下降（前 30 步内可见） | 不下降则数据/学习率有问题 |

冒烟测试结束后试听：`ckpts/child-tts_pinyin/model_30.pt` 合成一句，
**是可懂的语音（哪怕质量差）就算通过**；是纯噪声则停下来排查，不要进入正式训练。

```bash
# 把 speech_synthesis.py 里的 CKPT_FILE 改成要测的 checkpoint，然后：
python speech_synthesis.py "今天天气很好，我们去公园玩吧！"
# 或者用 infer_cli 直接指定 ckpt（见 notebook Step 2 的单元格）
```

## 2. 正式训练（一次跑完）

```bash
python -X utf8 src/f5_tts/train/finetune_cli.py \
    --epochs 50 --batch_size_per_gpu 4 \
    --learning_rate 1e-5 --scheduler cosine \
    --save_per_updates 500 --last_per_updates 100 \
    --keep_last_n_checkpoints -1 --logger tensorboard
```

参数依据：

- `batch_size_per_gpu 4`：本项目的自定义训练循环里单位是**样本数**（非官方的 frame 数）；
  220 条短音频（平均 ~3.8s）+ 8GB 显存，4 是稳妥值。OOM 就降回 2。
- `keep_last_n_checkpoints -1`：**保留所有中间 checkpoint**——一次训练也能选优的关键。
  （默认 0 会只留 model_last.pt。）
- 50 epochs × (224/4) ≈ 2800 步，auto warmup ≈ 280 步。
- 全程看 loss：`tensorboard --logdir runs`

## 3. Checkpoint 选优（训练后）

对几个候选 ckpt（如 model_1000 / 2000 / 2800 / model_last）分别：

1. 批量推理：修改 `batch_inference.py` 的 `CKPT_FILE`，跑出 `outputs/batch_inference/`
2. 音质指标：`python evaluate_quality.py`（PESQ/STOI/DNSMOS…）
3. 说话人相似度：`python evaluate_similarity.py`（SECS，voice cloning 的核心指标）

**选择标准：SECS 优先（像不像本人），DNSMOS 辅助（自不自然）。**
不要只看 loss——小数据集上 loss 最低的 ckpt 往往过拟合。

## 4. 已知坑（历史教训，详见 DEBUG_NOTES.md）

- 不要为消除报错而绕过权重加载逻辑（曾经 `load_checkpoint` 被改成 `return 0`）。
- 预训练 vocab 是**拼音** vocab，不是"英文 vocab"；推理脚本不要指向数据集字符 vocab。
- arrow 数据集里 text 必须是拼音 token 列表；是中文字符串就是错的。
