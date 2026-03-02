"""
TTS音频质量评估脚本 - 计算多种语音质量指标
TTS Audio Quality Evaluation Script - Calculate multiple speech quality metrics

评估指标：
- PESQ, STOI, CSIG, CBAK, COVL, DNSMOS, SRMR等
- 支持批量评估和详细报告生成
"""

import os
import sys
import json
import csv
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import glob
from tqdm import tqdm
import pprint

# 添加speechscore到路径
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "speechscore"))

from speechscore import SpeechScore

# 配置路径
TEST_DATA_DIR = ROOT_DIR / "data" / "test_data"
GENERATED_DIR = ROOT_DIR / "outputs" / "batch_inference"
RESULTS_DIR = ROOT_DIR / "evaluation_results"

# 创建结果目录
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(RESULTS_DIR / "evaluation.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 评估指标（排除帧对齐相关的指标如MCD）
EVALUATION_METRICS = [
    'PESQ',      # 感知语音质量 (1.0-4.5)
    'NB_PESQ',   # 窄带PESQ
    'STOI',      # 短时客观可懂度 (0.0-1.0)
    'SISDR',     # 信源失真比
    'FWSEGSNR',  # 分段信噪比
    'LSD',       # 对数谱距离
    'SNR',       # 信噪比
    'SSNR',      # 分段信噪比
    'CSIG',      # 信号失真（综合指标）(1.0-5.0)
    'CBAK',      # 背景噪声 (1.0-5.0)
    'COVL',      # 整体质量 (1.0-5.0)
    'DNSMOS',    # DNS挑战MOS分数
    'SRMR',      # 语音质量指标
]


def evaluate_single_pair(test_audio: Path, reference_audio: Path, evaluator: SpeechScore) -> Optional[Dict]:
    """
    评估单个音频对
    
    Args:
        test_audio: 生成的音频（测试音频）
        reference_audio: 参考音频（真实音频）
        evaluator: 评估器对象
    
    Returns:
        评估分数字典，失败返回None
    """
    try:
        # 尝试不同的采样率，因为不同音频可能有不同采样率
        # 优先尝试 24000（F5-TTS 默认），然后是 16000
        for sample_rate in [24000, 16000]:
            try:
                scores = evaluator(
                    test_path=str(test_audio),
                    reference_path=str(reference_audio),
                    window=None,
                    score_rate=sample_rate,  # F5-TTS 默认采样率是 24000
                    return_mean=False
                )
                if scores:  # 如果成功获取分数，返回
                    return scores
            except Exception as e:
                # 如果这个采样率失败，尝试下一个
                logger.debug(f"采样率 {sample_rate} 失败，尝试下一个: {str(e)}")
                continue
        
        # 如果所有采样率都失败，使用默认值再试一次
        scores = evaluator(
            test_path=str(test_audio),
            reference_path=str(reference_audio),
            window=None,
            score_rate=24000,
            return_mean=False
        )
        return scores
    except Exception as e:
        test_name = test_audio.name if isinstance(test_audio, (str, Path)) else str(test_audio)
        logger.error(f"评估失败 {test_name}: {str(e)}")
        return None


def extract_metric_value(value, metric_name: str) -> Optional[float]:
    """
    从不同格式的返回值中提取数值
    
    Args:
        value: 指标值（可能是字典、数值、列表等）
        metric_name: 指标名称
    
    Returns:
        提取的数值，失败返回None
    """
    if isinstance(value, (int, float)):
        return float(value)
    elif isinstance(value, dict):
        # DNSMOS返回字典，包含OVRL, SIG, BAK, P808_MOS等
        if metric_name == "DNSMOS":
            # 优先使用OVRL（整体质量），如果没有则尝试其他键
            if "OVRL" in value:
                return float(value["OVRL"])
            elif "mos" in value:
                return float(value["mos"])
            elif "P808_MOS" in value:
                return float(value["P808_MOS"])
            else:
                # 尝试所有可能的数值键
                for key in ["OVRL", "SIG", "BAK", "P808_MOS", "mos"]:
                    if key in value and isinstance(value[key], (int, float)):
                        return float(value[key])
        elif "mean" in value:
            return float(value["mean"])
        elif "score" in value:
            return float(value["score"])
        else:
            # 取第一个数值
            for v in value.values():
                if isinstance(v, (int, float)):
                    return float(v)
    elif isinstance(value, (list, tuple)) and len(value) > 0:
        # 如果是列表，取平均值
        numeric_values = [v for v in value if isinstance(v, (int, float))]
        if numeric_values:
            return sum(numeric_values) / len(numeric_values)
    
    return None


def evaluate_batch() -> Dict:
    """
    批量评估
    
    Returns:
        评估结果字典
    """
    logger.info("=" * 60)
    logger.info("TTS音频质量评估 - 开始评估")
    logger.info("=" * 60)
    
    # 检查目录
    if not GENERATED_DIR.exists():
        logger.error(f"生成音频目录不存在: {GENERATED_DIR}")
        logger.error("请先运行批量推理脚本！")
        return {}
    
    # 初始化评估器
    logger.info(f"初始化评估指标: {', '.join(EVALUATION_METRICS)}")
    evaluator = SpeechScore(EVALUATION_METRICS)
    
    # 获取所有测试音频文件
    test_files = sorted([Path(f) for f in glob.glob(str(TEST_DATA_DIR / "*.wav"))])
    
    if not test_files:
        logger.error(f"在 {TEST_DATA_DIR} 中未找到WAV文件")
        return {}
    
    # 匹配生成的文件
    generated_files = {}
    for test_file in test_files:
        test_name = test_file.stem
        generated_file = GENERATED_DIR / f"{test_name}_generated.wav"
        if generated_file.exists():
            generated_files[test_file] = generated_file
        else:
            logger.warning(f"未找到对应的生成文件: {generated_file}")
    
    if not generated_files:
        logger.error("未找到任何匹配的生成文件")
        return {}
    
    logger.info(f"找到 {len(generated_files)} 对音频文件进行评估")
    logger.info("-" * 60)
    
    # 批量评估
    all_results = {}
    successful_evaluations = 0
    
    for test_audio, generated_audio in tqdm(generated_files.items(), desc="评估进度"):
        test_name = test_audio.stem
        scores = evaluate_single_pair(generated_audio, test_audio, evaluator)
        
        if scores:
            all_results[test_name] = {
                "test_audio": str(test_audio),
                "generated_audio": str(generated_audio),
                "scores": scores,
                "timestamp": datetime.now().isoformat()
            }
            successful_evaluations += 1
    
    # 计算平均分数
    if all_results:
        mean_scores = {}
        for metric in EVALUATION_METRICS:
            metric_values = []
            for result in all_results.values():
                if metric in result["scores"]:
                    value = extract_metric_value(result["scores"][metric], metric)
                    if value is not None:
                        metric_values.append(value)
            
            if metric_values:
                mean_scores[metric] = {
                    "mean": sum(metric_values) / len(metric_values),
                    "min": min(metric_values),
                    "max": max(metric_values),
                    "std": (sum((x - sum(metric_values) / len(metric_values))**2 for x in metric_values) / len(metric_values))**0.5 if len(metric_values) > 1 else 0,
                    "count": len(metric_values)
                }
        
        all_results["Mean_Score"] = mean_scores
        all_results["Summary"] = {
            "total_pairs": len(generated_files),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": len(generated_files) - successful_evaluations,
            "evaluation_time": datetime.now().isoformat()
        }
    
    # 保存结果
    results_file = RESULTS_DIR / "evaluation_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    # 打印结果摘要
    logger.info("-" * 60)
    logger.info(f"评估完成！")
    logger.info(f"成功评估: {successful_evaluations}/{len(generated_files)}")
    logger.info(f"结果已保存: {results_file}")
    logger.info("-" * 60)
    logger.info("平均分数摘要:")
    if "Mean_Score" in all_results:
        pprint.pprint(all_results["Mean_Score"], width=100)
    
    # 生成CSV报告
    generate_csv_report(all_results, RESULTS_DIR / "evaluation_report.csv")
    
    # 生成Markdown报告
    generate_markdown_report(all_results, RESULTS_DIR / "evaluation_report.md")
    
    return all_results


def generate_csv_report(results: Dict, csv_path: Path):
    """生成CSV格式的评估报告"""
    if "Mean_Score" not in results:
        return
    
    # 提取所有指标
    metrics = list(results["Mean_Score"].keys())
    
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        
        # 写入表头
        header = ["Metric", "Mean", "Min", "Max", "Std", "Count"]
        writer.writerow(header)
        
        # 写入数据
        for metric in metrics:
            mean_data = results["Mean_Score"][metric]
            writer.writerow([
                metric,
                f"{mean_data['mean']:.4f}",
                f"{mean_data['min']:.4f}",
                f"{mean_data['max']:.4f}",
                f"{mean_data['std']:.4f}",
                mean_data['count']
            ])
    
    logger.info(f"CSV报告已保存: {csv_path}")


def generate_markdown_report(results: Dict, md_path: Path):
    """生成Markdown格式的评估报告"""
    if "Mean_Score" not in results:
        return
    
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# TTS音频质量评估报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        if "Summary" in results:
            summary = results["Summary"]
            f.write("## 评估摘要\n\n")
            f.write(f"- 总音频对数: {summary['total_pairs']}\n")
            f.write(f"- 成功评估: {summary['successful_evaluations']}\n")
            f.write(f"- 失败评估: {summary['failed_evaluations']}\n\n")
        
        f.write("## 平均分数\n\n")
        f.write("| 指标 | 平均值 | 最小值 | 最大值 | 标准差 | 样本数 |\n")
        f.write("|------|--------|--------|--------|--------|--------|\n")
        
        for metric, stats in results["Mean_Score"].items():
            f.write(f"| {metric} | {stats['mean']:.4f} | {stats['min']:.4f} | "
                   f"{stats['max']:.4f} | {stats['std']:.4f} | {stats['count']} |\n")
        
        f.write("\n## 关键指标说明\n\n")
        f.write("- **PESQ**: 感知语音质量 (1.0-4.5，越高越好)\n")
        f.write("- **STOI**: 短时客观可懂度 (0.0-1.0，越高越好)\n")
        f.write("- **CSIG/CBAK/COVL**: 综合质量指标 (1.0-5.0，越高越好)\n")
        f.write("- **DNSMOS**: DNS挑战MOS分数 (越高越好)\n")
        f.write("- **SRMR**: 语音质量指标 (越高越好)\n")
    
    logger.info(f"Markdown报告已保存: {md_path}")


if __name__ == "__main__":
    evaluate_batch()


