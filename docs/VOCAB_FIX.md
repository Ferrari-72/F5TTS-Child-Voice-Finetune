# Vocab文件修复说明

## 🐛 问题发现

生成的语音"不是人话"的原因是**使用了错误的vocab文件**！

### 问题分析

**错误使用的vocab：**
- `ckpts/child-tts/vocab.txt` - 预训练模型的vocab
- 内容：英文、数字、标点符号、拼音（a1, ai1, ai2等）
- **不适合中文文本！**

**应该使用的vocab：**
- `data/child-tts_processed/vocab.txt` - 数据处理时生成的中文vocab
- 内容：中文标点、中文汉字（一、七、上、下、了等）
- **专门为中文数据集生成！**

## ✅ 已修复的文件

1. `tts_batch_inference.py` ✅
2. `inference_quick_test.py` ✅
3. `hyperparameter_optimization.py` ✅

## 📝 修复内容

所有脚本中的vocab路径已从：
```python
VOCAB_FILE = ROOT_DIR / "ckpts" / "child-tts" / "vocab.txt"  # ❌ 错误
```

改为：
```python
VOCAB_FILE = ROOT_DIR / "data" / "child-tts_processed" / "vocab.txt"  # ✅ 正确
```

## 🔍 如何验证

### 检查vocab文件内容

**预训练vocab（错误）：**
```bash
head -20 ckpts/child-tts/vocab.txt
# 输出：空白、!、"、#、$、%、&、'、(、)、*、+、,、-、.、/、0、1、2、3...
```

**中文vocab（正确）：**
```bash
head -20 data/child-tts_processed/vocab.txt
# 输出：!、,、?、。、一、七、上、下、不、世、东、两、个、中、为、久、么、义、之、也...
```

### 重新生成音频

修复后，重新运行推理：

```bash
# 快速测试
python inference_quick_test.py

# 批量推理
python tts_batch_inference.py
```

现在生成的语音应该能正确说中文了！

## ⚠️ 注意事项

1. **Notebook中的引用**：如果notebook中还有旧的vocab路径，需要手动更新
2. **配置文件**：检查 `src/f5_tts/infer/examples/basic/basic.toml` 等配置文件
3. **其他脚本**：确保所有使用vocab的脚本都已更新

## 📚 相关文件

- `data/child-tts_processed/vocab.txt` - 正确的中文vocab（465个字符）
- `ckpts/child-tts/vocab.txt` - 预训练vocab（2546个token，包含拼音）


