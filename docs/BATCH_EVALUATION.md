# 批量推理与评估使用指南

## 概述

本项目提供了完整的批量推理和评估流程，用于：
1. 对测试数据进行批量语音合成
2. 评估生成音频的质量
3. 生成详细的评估报告

## 文件说明

- `tts_batch_inference.py`: 批量推理脚本，对测试数据进行语音合成
- `tts_evaluation.py`: 音频质量评估脚本，计算多种质量指标
- `optimize_inference_params.py`: 推理参数优化脚本（可选）
- `setup_dependencies.py`: 安装评估依赖的辅助脚本

## 快速开始

### 1. 安装依赖

```bash
python setup_dependencies.py
```

或者在notebook中运行相应的单元格。

### 2. 批量推理

运行批量推理脚本，对所有测试数据进行语音合成：

```bash
python batch_inference.py
```

**配置说明：**
- 测试数据目录：`data/test_data/`
- 输出目录：`test_outputs/batch_inference/`
- 参考音频：`data/child-tts/000018.wav`（用于音色克隆）
- 推理参数：
  - `nfe_step`: 100（采样步数，越高质量越好但速度越慢）
  - `cfg_strength`: 2.5（引导强度，2.0-3.0之间）

### 3. 评估音频质量

运行评估脚本，计算音频质量分数：

```bash
python tts_evaluation.py
```

**评估指标：**
- **PESQ**: 感知语音质量（1.0-4.5，越高越好）
- **STOI**: 短时客观可懂度（0.0-1.0，越高越好）
- **CSIG, CBAK, COVL**: 综合质量指标（1.0-5.0，越高越好）
- **DNSMOS**: DNS挑战MOS分数
- **SRMR**: 语音质量指标

> **注意：** 排除MCD（梅尔倒谱距离），因为它需要帧对齐。

### 4. 查看结果

评估结果保存在：
- JSON格式：`evaluation_results/evaluation_results.json`
- CSV格式：`evaluation_results/evaluation_report.csv`

## 在Notebook中使用

完整的流程已集成到 `AIAA2205_Assignment_2_TTS_Finetuning.ipynb` 中：

1. **Cell 41-42**: 安装依赖
2. **Cell 43-44**: 批量推理
3. **Cell 45-46**: 评估音频质量
4. **Cell 47-48**: 查看评估结果

## 参数优化（可选）

如果需要优化推理参数以获得更高的分数，可以运行：

```bash
python optimize_inference_params.py
```

该脚本会测试不同的参数组合，并推荐最佳配置。

## 故障排除

### 问题1: pesq模块未找到

```bash
pip install pesq==0.0.4
```

### 问题2: CUDA环境变量错误（Windows）

脚本已自动处理Windows下的CUDA环境变量设置，无需手动配置。

### 问题3: 路径错误

确保在项目根目录下运行脚本：
```bash
cd F:/AI-project/F5TTS/AIAA2205-assignment2-F5-TTS
python batch_inference.py
```

### 问题4: 模型检查点不存在

确保已完成微调训练，检查点应位于：
```
ckpts/child-tts_processed/model_last.pt
```

## 输出文件结构

```
test_outputs/
├── batch_inference/
│   ├── YZ-0002_generated.wav
│   ├── YZ-0003_generated.wav
│   ├── ...
│   └── metadata.json
│
evaluation_results/
├── evaluation_results.json
└── evaluation_report.csv
```

## 注意事项

1. **推理时间**: 批量推理可能需要较长时间，取决于测试数据数量和GPU性能
2. **内存使用**: 确保有足够的GPU内存（建议8GB+）
3. **评估时间**: 评估过程也需要一定时间，请耐心等待
4. **分数解释**: 不同指标有不同的范围，请参考各指标的说明

## 优化建议

为了提高评估分数，可以尝试：

1. **增加采样步数** (`nfe_step`): 50 → 100 → 150（质量提升但速度下降）
2. **调整引导强度** (`cfg_strength`): 2.0 → 2.5 → 3.0（更忠实于参考音频）
3. **使用更长的参考音频**: 选择质量更好的参考音频
4. **增加训练轮数**: 如果分数仍然较低，可以增加微调训练的轮数

## 联系与支持

如有问题，请检查：
1. 依赖是否正确安装
2. 模型检查点是否存在
3. 测试数据路径是否正确
4. GPU是否可用

