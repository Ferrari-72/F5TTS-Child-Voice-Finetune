# 项目结构说明

## 📁 目录结构

```
AIAA2205-assignment2-F5-TTS/
├── data/                          # 数据目录
│   ├── child-tts/                # 原始训练数据（220个儿童语音文件）
│   ├── child-tts_processed/      # 预处理后的数据（Arrow格式）
│   ├── test_data/                # 测试数据（60+个测试音频）
│   └── dialect_data/             # 方言数据（可选）
│
├── ckpts/                        # 模型检查点目录
│   ├── child-tts/                # 预训练模型
│   │   ├── pretrained_model_1250000.safetensors
│   │   └── vocab.txt
│   └── child-tts_processed/       # 微调后的模型
│       └── model_last.pt
│
├── outputs/                      # 输出目录
│   └── batch_inference/          # 批量推理结果
│       ├── *_generated.wav       # 生成的音频文件
│       ├── inference_progress.json  # 进度文件
│       ├── inference_metadata.json  # 元数据
│       └── inference.log         # 日志文件
│
├── evaluation_results/           # 评估结果目录
│   ├── evaluation_results.json   # JSON格式结果
│   ├── evaluation_report.csv     # CSV格式报告
│   ├── evaluation_report.md      # Markdown格式报告
│   └── evaluation.log            # 评估日志
│
├── docs/                         # 文档目录
│   ├── SCORE_OPTIMIZATION_METHODS.md  # 分数优化方法
│   ├── OPTIMIZATION_SUMMARY.md        # 优化总结
│   ├── BATCH_EVALUATION.md            # 批量评估指南
│   ├── PROJECT_SUMMARY.md             # 项目总结
│   └── PROJECT_STRUCTURE.md           # 项目结构说明（本文件）
│
├── src/                          # 源代码目录
│   └── f5_tts/                   # F5-TTS 核心代码
│       ├── configs/              # 模型配置文件
│       ├── infer/                 # 推理相关代码
│       ├── train/                 # 训练相关代码
│       └── Models/                # 模型定义
│
├── speechscore/                  # 评估工具库
│   ├── scores/                   # 各种评估指标实现
│   └── speechscore.py            # 主评估接口
│
├── tts_batch_inference.py        # 批量推理脚本（主脚本）
├── tts_evaluation.py             # 音频质量评估脚本
├── hyperparameter_optimization.py # 参数优化脚本
├── inference_quick_test.py       # 快速测试脚本
├── setup_dependencies.py         # 依赖安装脚本
│
├── AIAA2205_Assignment_2_TTS_Finetuning.ipynb  # 主Notebook
├── requirements.txt              # Python依赖
├── README.md                     # 项目说明
└── LICENSE                       # 许可证
```

## 📝 主要文件说明

### 核心脚本

1. **tts_batch_inference.py**
   - 批量推理主脚本
   - 支持进度保存和恢复
   - 包含音频后处理
   - 详细的日志记录

2. **tts_evaluation.py**
   - 音频质量评估脚本
   - 支持多种评估指标
   - 生成多种格式的报告

3. **hyperparameter_optimization.py**
   - 推理参数优化工具
   - 自动搜索最佳参数组合

4. **inference_quick_test.py**
   - 快速测试工具
   - 用于验证推理流程

5. **setup_dependencies.py**
   - 依赖安装脚本
   - 安装评估所需的包

### 配置文件

- **requirements.txt**: Python依赖包列表
- **src/f5_tts/configs/F5TTS_v1_Base.yaml**: 模型配置文件

### 文档文件

所有文档位于 `docs/` 目录：
- **SCORE_OPTIMIZATION_METHODS.md**: 详细的分数优化方法
- **OPTIMIZATION_SUMMARY.md**: 优化总结报告
- **BATCH_EVALUATION.md**: 批量评估使用指南
- **PROJECT_SUMMARY.md**: 项目完成总结

## 🔄 工作流程

1. **数据准备** → `data/child-tts/`
2. **数据预处理** → `data/child-tts_processed/`
3. **模型微调** → `ckpts/child-tts_processed/`
4. **批量推理** → `outputs/batch_inference/`
5. **质量评估** → `evaluation_results/`

## 📊 输出文件说明

### 推理输出

- `*_generated.wav`: 生成的音频文件
- `inference_progress.json`: 进度保存文件（支持中断后继续）
- `inference_metadata.json`: 完整的元数据信息
- `inference.log`: 详细的日志记录

### 评估输出

- `evaluation_results.json`: 完整的评估结果（JSON格式）
- `evaluation_report.csv`: 评估报告（CSV格式，便于Excel打开）
- `evaluation_report.md`: 评估报告（Markdown格式，便于阅读）
- `evaluation.log`: 评估日志

## 🎯 命名规范

### 文件命名

- 脚本文件：`tts_*.py` 或 `*_*.py`（使用下划线分隔）
- 配置文件：`*.yaml`, `*.toml`, `*.json`
- 文档文件：`*.md`（位于 `docs/` 目录）

### 目录命名

- 使用小写字母和下划线
- 描述性名称（如 `batch_inference`, `evaluation_results`）

## 🔧 维护建议

1. **定期清理输出目录**：避免占用过多磁盘空间
2. **备份重要文件**：模型检查点、评估结果等
3. **更新文档**：代码变更时同步更新文档
4. **版本控制**：使用 Git 管理代码版本


