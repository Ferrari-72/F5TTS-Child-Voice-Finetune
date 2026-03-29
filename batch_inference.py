"""
TTS批量推理脚本 - 对测试数据进行高质量语音合成
TTS Batch Inference Script - High-quality speech synthesis for test data

功能特性：
- 优化的推理参数（nfe_step=150, cfg_strength=3.0）
- 进度保存和恢复（支持中断后继续）
- 音频后处理（音量归一化、静音去除）
- 详细的日志记录
- 错误处理和重试机制
"""

import os
import sys
import io
import subprocess
import glob
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict
import whisper
from tqdm import tqdm
import soundfile as sf
import numpy as np

# 修复Windows控制台编码问题
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
    except Exception:
        pass

# 项目根目录
ROOT_DIR = Path(__file__).parent.absolute()
os.chdir(ROOT_DIR)

# 配置路径
TEST_DATA_DIR = ROOT_DIR / "data" / "test_data"
OUTPUT_DIR = ROOT_DIR / "outputs" / "batch_inference"
REF_AUDIO = ROOT_DIR / "data" / "child-tts" / "000018.wav"
REF_TEXT = "城里有好多游乐场，可好玩儿了！"

# 模型路径
MODEL_CFG = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"
CKPT_FILE = ROOT_DIR / "ckpts" / "child-tts_processed" / "model_last.pt"
VOCAB_FILE = ROOT_DIR / "data" / "child-tts_processed" / "vocab.txt"  # 使用中文vocab，不是预训练的英文vocab

# 优化的推理参数
NFE_STEP = 150  # 采样步数（50-200，更高=更好质量但更慢）
CFG_STRENGTH = 3.0  # 引导强度（2.0-3.5，更高=更忠实于参考音频）
SWAY_SAMPLING_COEF = -1.0  # Sway采样系数（-1表示使用默认值）
REMOVE_SILENCE = True  # 是否移除长静音段

# 进度保存文件
PROGRESS_FILE = OUTPUT_DIR / "inference_progress.json"

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(OUTPUT_DIR / "inference.log", encoding="utf-8"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# 全局Whisper模型（避免重复加载）
_whisper_model = None


def transcribe_audio(audio_path: Path, model_size: str = "large-v2") -> str:
    """使用Whisper转录音频"""
    global _whisper_model
    if _whisper_model is None:
        logger.info("加载Whisper模型...")
        _whisper_model = whisper.load_model(model_size)

    result = _whisper_model.transcribe(str(audio_path), language="zh")
    return result["text"].strip()


def post_process_audio(audio_path: Path, target_rms: float = 0.1) -> bool:
    """
    音频后处理：音量归一化

    Args:
        audio_path: 音频文件路径
        target_rms: 目标RMS值

    Returns:
        是否处理成功
    """
    try:
        audio, sr = sf.read(str(audio_path))

        # 计算当前RMS
        current_rms = np.sqrt(np.mean(audio**2))

        # 如果RMS太小，进行归一化
        if current_rms < target_rms:
            scale_factor = target_rms / current_rms
            audio = audio * scale_factor
            # 防止削波
            audio = np.clip(audio, -1.0, 1.0)
            sf.write(str(audio_path), audio, sr)
            logger.debug(f"音频归一化完成: {audio_path.name}")

        return True
    except Exception as e:
        logger.warning(f"音频后处理失败 {audio_path.name}: {str(e)}")
        return False


def load_progress() -> Dict:
    """加载进度文件"""
    if PROGRESS_FILE.exists():
        try:
            with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"加载进度文件失败: {str(e)}")
    return {"processed_files": [], "failed_files": [], "last_update": None}


def save_progress(progress: Dict):
    """保存进度文件"""
    progress["last_update"] = datetime.now().isoformat()
    try:
        with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存进度文件失败: {str(e)}")


def run_inference(test_audio_path: Path, output_name: str, retry_count: int = 2) -> bool:
    """
    运行单次推理

    Args:
        test_audio_path: 测试音频路径
        output_name: 输出文件名
        retry_count: 重试次数

    Returns:
        是否成功
    """
    for attempt in range(retry_count + 1):
        try:
            # 先转录测试音频获取文本
            gen_text = transcribe_audio(test_audio_path)
            logger.info(f"转录文本: {gen_text[:50]}...")

            output_path = OUTPUT_DIR / output_name

            # 构建推理命令
            cmd = [
                sys.executable,
                str(ROOT_DIR / "src" / "f5_tts" / "infer" / "infer_cli.py"),
                "--model_cfg",
                str(MODEL_CFG),
                "--ckpt_file",
                str(CKPT_FILE),
                "--vocab_file",
                str(VOCAB_FILE),
                "--ref_audio",
                str(REF_AUDIO),
                "--ref_text",
                REF_TEXT,
                "--gen_text",
                gen_text,
                "--output_dir",
                str(OUTPUT_DIR),
                "--output_file",
                output_name,
                "--nfe_step",
                str(NFE_STEP),
                "--cfg_strength",
                str(CFG_STRENGTH),
            ]

            # 添加可选参数
            if SWAY_SAMPLING_COEF > 0:
                cmd.extend(["--sway_sampling_coef", str(SWAY_SAMPLING_COEF)])

            if REMOVE_SILENCE:
                cmd.append("--remove_silence")

            # 设置CUDA设备（Windows兼容）
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = "0"

            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=str(ROOT_DIR),
                timeout=600,  # 10分钟超时（因为nfe_step=150需要更长时间）
            )

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else result.stdout
                test_name = test_audio_path.name
                logger.warning(f"推理失败 {test_name} (尝试 {attempt + 1}/{retry_count + 1}): {error_msg[:200]}")

                if attempt < retry_count:
                    continue
                return False
            else:
                # 验证输出文件是否存在
                if output_path.exists() and output_path.stat().st_size > 0:
                    # 音频后处理
                    post_process_audio(output_path)
                    logger.info(f"成功生成: {output_name}")
                    return True
                else:
                    logger.warning(f"输出文件不存在或为空: {output_name}")
                    if attempt < retry_count:
                        continue
                    return False

        except subprocess.TimeoutExpired:
            test_name = test_audio_path.name
            logger.error(f"推理超时: {test_name} (尝试 {attempt + 1}/{retry_count + 1})")
            if attempt < retry_count:
                continue
            return False
        except Exception as e:
            test_name = test_audio_path.name
            logger.error(f"推理异常 {test_name} (尝试 {attempt + 1}/{retry_count + 1}): {str(e)}")
            if attempt < retry_count:
                continue
            return False

    return False


def main():
    """主函数：批量推理"""
    logger.info("=" * 60)
    logger.info("TTS批量推理脚本 - 开始处理测试数据")
    logger.info("=" * 60)

    # 检查路径
    if not TEST_DATA_DIR.exists():
        logger.error(f"测试数据目录不存在: {TEST_DATA_DIR}")
        return

    if not CKPT_FILE.exists():
        logger.error(f"模型检查点不存在: {CKPT_FILE}")
        logger.error("请先运行微调训练！")
        return

    # 获取所有测试音频文件
    test_files = sorted([Path(f) for f in glob.glob(str(TEST_DATA_DIR / "*.wav"))])

    if not test_files:
        logger.error(f"在 {TEST_DATA_DIR} 中未找到WAV文件")
        return

    logger.info(f"找到 {len(test_files)} 个测试音频文件")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"推理参数: nfe_step={NFE_STEP}, cfg_strength={CFG_STRENGTH}")
    logger.info("-" * 60)

    # 加载进度
    progress = load_progress()
    processed_set = set(progress.get("processed_files", []))
    failed_set = set(progress.get("failed_files", []))

    # 保存元数据
    metadata = {
        "total_files": len(test_files),
        "processed_files": [],
        "failed_files": [],
        "config": {
            "nfe_step": NFE_STEP,
            "cfg_strength": CFG_STRENGTH,
            "sway_sampling_coef": SWAY_SAMPLING_COEF,
            "remove_silence": REMOVE_SILENCE,
            "ref_audio": str(REF_AUDIO),
            "ref_text": REF_TEXT,
        },
        "start_time": datetime.now().isoformat(),
    }

    # 批量处理
    success_count = 0
    skipped_count = 0

    for test_file in tqdm(test_files, desc="处理进度"):
        test_name = test_file.stem
        output_name = f"{test_name}_generated.wav"

        # 检查是否已处理
        if test_name in processed_set:
            logger.debug(f"跳过已处理文件: {test_name}")
            skipped_count += 1
            continue

        # 检查输出文件是否已存在
        output_path = OUTPUT_DIR / output_name
        if output_path.exists() and output_path.stat().st_size > 0:
            logger.debug(f"输出文件已存在，跳过: {output_name}")
            processed_set.add(test_name)
            skipped_count += 1
            continue

        # 运行推理
        if run_inference(test_file, output_name):
            success_count += 1
            processed_set.add(test_name)
            if test_name in failed_set:
                failed_set.remove(test_name)

            metadata["processed_files"].append(
                {"input": str(test_file), "output": str(output_path), "timestamp": datetime.now().isoformat()}
            )

            # 每处理10个文件保存一次进度
            if success_count % 10 == 0:
                progress["processed_files"] = list(processed_set)
                progress["failed_files"] = list(failed_set)
                save_progress(progress)
        else:
            failed_set.add(test_name)
            metadata["failed_files"].append({"input": str(test_file), "timestamp": datetime.now().isoformat()})

    # 保存最终元数据
    metadata["end_time"] = datetime.now().isoformat()
    metadata["success_count"] = success_count
    metadata["failed_count"] = len(failed_set)
    metadata["skipped_count"] = skipped_count

    metadata_path = OUTPUT_DIR / "inference_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 保存最终进度
    progress["processed_files"] = list(processed_set)
    progress["failed_files"] = list(failed_set)
    save_progress(progress)

    logger.info("-" * 60)
    logger.info("批量推理完成！")
    logger.info(f"成功: {success_count}/{len(test_files)}")
    logger.info(f"跳过: {skipped_count}")
    logger.info(f"失败: {len(failed_set)}")
    logger.info(f"元数据已保存: {metadata_path}")
    logger.info(f"日志已保存: {OUTPUT_DIR / 'inference.log'}")


if __name__ == "__main__":
    main()
