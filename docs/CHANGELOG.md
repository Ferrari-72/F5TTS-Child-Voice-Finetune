# 更新日志

## [2.0.0] - 2025-11-17

### 🎉 重大更新

#### 代码优化
- ✅ **改进的批量推理脚本** (`tts_batch_inference.py`)
  - 添加进度保存和恢复功能（支持中断后继续）
  - 添加音频后处理（音量归一化）
  - 改进的错误处理和重试机制
  - 详细的日志记录
  - 优化推理参数（nfe_step=150, cfg_strength=3.0）

- ✅ **改进的评估脚本** (`tts_evaluation.py`)
  - 改进的采样率处理（支持24000/16000）
  - 优化的DNSMOS分数提取逻辑
  - 生成多种格式报告（JSON, CSV, Markdown）
  - 详细的日志记录

#### 文件重命名（更正式的项目结构）
- `batch_inference.py` → `tts_batch_inference.py`
- `evaluate_audio.py` → `tts_evaluation.py`
- `install_dependencies.py` → `setup_dependencies.py`
- `optimize_inference_params.py` → `hyperparameter_optimization.py`
- `quick_test.py` → `inference_quick_test.py`
- `requirement.txt` → `requirements.txt`

#### 文档整理
- ✅ 创建 `docs/` 目录
- ✅ 移动所有文档到 `docs/` 目录
- ✅ 更新所有文档中的文件引用
- ✅ 创建 `PROJECT_STRUCTURE.md` 项目结构说明
- ✅ 创建 `CHANGELOG.md` 更新日志
- ✅ 更新 `README.md` 主文档

#### 目录结构优化
- `test_outputs/` → `outputs/`（更标准的命名）

### 📊 性能提升

- **推理质量**: 通过优化参数，预期所有评估指标提升 0.1-0.3 分
- **可靠性**: 添加进度保存，支持长时间运行的批量推理
- **可维护性**: 改进的代码结构和文档

### 🔧 技术改进

- 添加日志系统（logging）
- 改进错误处理
- 支持音频后处理
- 更好的进度跟踪

## [1.0.0] - 初始版本

### 初始功能
- 批量推理脚本
- 音频质量评估脚本
- 参数优化工具
- 基础文档


