# 问题排查笔记（DEBUG NOTES）

> 记录本项目微调失败（输出纯噪声）的完整排查过程和发现的错误。
> 结论：**不是一个 bug，而是四个 bug 叠加**，任何一个都足以让微调失败。

## 一、最终诊断：四个叠加的致命错误

### Bug 1（最大根因）：预训练权重从未被加载
- 位置：`src/f5_tts/Models/Trainer.py:185`
- 现象：`load_checkpoint()` 被改成 `return 0  # 直接返回0，不加载任何旧checkpoint，彻底规避属性冲突`
- 后果：所谓"微调"实际是**从随机初始化在 14 分钟数据上从头训练**。无论 vocab 对不对，输出都必然是噪声。
- 正确行为（官方 F5-TTS）：`load_checkpoint()` 按 `model_last.pt` → 最新 `model_N.pt` → `pretrained_*.safetensors` 的优先级加载权重，safetensors 预训练文件加载进 EMA 和 online model。
- 教训：不要为了让报错消失而"绕过"加载逻辑——属性冲突应该去查冲突的 key 是什么。

### Bug 2：训练用错 vocab（README 已记录的那个）
- 训练用了数据准备自动生成的 464 字中文 vocab（`data/child-tts_processed/vocab.txt`），
  而预训练模型是 2545-token 拼音 vocab（`ckpts/child-tts/vocab.txt`）。
- 后果：text embedding 维度 464 vs 2545 对不上，即使加载预训练权重也会破坏文本嵌入层。

### Bug 3（藏得最深）：数据准备阶段没有做拼音转换
- 官方流程：`prepare_csv_wavs.py:87` 在写入 arrow 数据集**之前**用
  `convert_char_to_pinyin()` 把中文文本转成拼音 token 列表（如 `["cheng2", "li3", ...]`），
  训练时按音节查 2545-token 拼音 vocab。
- 本项目：`data_prepare.py` 把 Whisper 转写的**原始中文**直接存进 arrow；
  训练时 `list_str_to_idx` 逐字查表，中文字符在拼音 vocab 里全部 OOV → 映射为 0。
- 后果：用 2545 拼音 vocab 训练时，模型看到的文本序列全 0，等于无条件生成。
- 验证方法：读 `raw.arrow` 的 text 列，如果是中文字符串（而不是拼音列表），就是错的。
  `data/child-tts_processed_pinyin/` 名不副实——里面 text 仍是中文字符串，vocab 也是 475 个中文字。

### Bug 4：`docs/VOCAB_FIX.md` 的"修复"方向反了（该文件已删除）
- 该文档让推理脚本（batch_inference / quick_inference / speech_synthesis / hparam_search）
  改用 464 字数据集 vocab，注释写着"使用中文vocab，不是预训练的英文vocab"——这是误解：
  预训练 vocab 不是"英文 vocab"，是**拼音 vocab**。
- 推理时 `utils_infer.py` 会把输入文本转成拼音 token，再到 vocab 里查；
  用 464 中文字 vocab 时拼音 token 全部 OOV → 即使 checkpoint 是好的也输出垃圾。
- 正确方向：**训练、推理统一使用预训练 2545-token 拼音 vocab**（`generate_samples.py` 是唯一用对的脚本）。

## 二、次级问题

| 问题 | 位置 | 影响 |
|---|---|---|
| `keep_last_n_checkpoints=0` | finetune_cli.py 默认值 | 中间 checkpoint 全部不保存，只有 model_last.pt，无法选优。只能训一次的项目必须 `-1`（全保留） |
| 硬编码绝对路径 `F:/AI-project/...` | finetune_cli.py:4,67,76 等 | 换机器/换目录即坏，应改为相对仓库根目录 |
| batch_size=2 | finetune_cli.py | 本项目自定义训练循环里是**样本数**（不是官方的 frame 数），不是 bug 但偏小；短音频 + 8GB 显存可到 4~8 |
| `--num_warmup_updates -1` 自动 warmup | finetune_cli.py:273 | 本项目自己加的逻辑，经核实**是对的**（官方默认 20000 超过小数据集总步数会导致 LR 始终接近 0） |
| 各文档中 vocab 数字不一致 | README/docs | 464 / 465 / 2545 / 9327 / 9331 混用，实测：预训练=2545 行，数据集=464 行 |

## 三、一次性训练的成功链（修复后的正确姿势）

1. **数据**：复用已有 Whisper 转写，写脚本把 arrow 里的中文文本转拼音 token 列表重新保存
   （不必重跑 Whisper），得到 `data/child-tts_pinyin/`（text 为 list<string>）。
2. **训练**：恢复 Trainer 权重加载；`tokenizer=custom` + `tokenizer_path=<预训练vocab.txt>`；
   `keep_last_n_checkpoints=-1`；确认日志打印 `vocab : 2545`。
3. **冒烟测试**（正式训练前，约 5 分钟）：跑 ~50 步 → 确认 loss 下降 →
   用中间 ckpt 合成一句话，确认是可懂语音（哪怕质量差）。
4. **正式训练**后：对多个中间 ckpt 分别跑 DNSMOS + SECS（说话人相似度），选最好的。
5. **推理**：所有脚本统一指向预训练 vocab.txt。

## 四、环境备忘

- 原始工程（含数据、ckpts、runs）在 `F:/AI-project/F5TTS/AIAA2205-assignment2-F5-TTS/`
- 本仓库是 GitHub 发布版，路径 `F:/berkeley/eecs127/F5TTS-Child-Voice-Finetune-main`
- 预训练 vocab 实测 2545 行（拼音音节 + ASCII 字符）；数据集自动 vocab 464 行（中文字符）
