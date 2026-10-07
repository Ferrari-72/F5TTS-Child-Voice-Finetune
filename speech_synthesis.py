"""
语音生成脚本 - 快速生成高质量中文语音
Speech Generation Script - Quick generation of high-quality Chinese speech

使用方法:
    python generate_speech.py "要生成的文本"
    或
    python generate_speech.py  # 使用默认文本
"""

import os
import sys
import io
import subprocess
from pathlib import Path
from datetime import datetime

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
OUTPUT_DIR = ROOT_DIR / "outputs" / "generated_speech"
REF_AUDIO = ROOT_DIR / "data" / "child-tts" / "000018.wav"
REF_TEXT = "城里有好多游乐场，可好玩儿了！"  # 参考音频的文本

# 模型路径（已修复vocab路径）
MODEL_CFG = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"
CKPT_FILE = ROOT_DIR / "ckpts" / "child-tts_pinyin" / "model_last.pt"
VOCAB_FILE = ROOT_DIR / "ckpts" / "child-tts" / "vocab.txt"  # pretrained pinyin vocab (2545 tokens); the dataset-generated char vocab breaks pinyin tokenization

# 优化的推理参数（高质量设置）
NFE_STEP = 150  # 采样步数（高质量）
CFG_STRENGTH = 3.0  # 引导强度（更忠实于参考音频）
REMOVE_SILENCE = True  # 移除长静音段

# 创建输出目录
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def generate_speech(text: str, output_name: str = None) -> Path:
    """
    生成语音

    Args:
        text: 要生成的文本
        output_name: 输出文件名（可选）

    Returns:
        生成的音频文件路径
    """
    if output_name is None:
        # 使用时间戳生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"speech_{timestamp}.wav"

    output_path = OUTPUT_DIR / output_name

    print("=" * 60)
    print("语音生成 - 使用优化参数")
    print("=" * 60)
    print(f"生成文本: {text}")
    print(f"参考音频: {REF_AUDIO.name}")
    print(f"参考文本: {REF_TEXT}")
    print(f"输出文件: {output_name}")
    print(f"推理参数: nfe_step={NFE_STEP}, cfg_strength={CFG_STRENGTH}")
    print("-" * 60)

    # 构建推理命令
    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "f5_tts" / "infer" / "infer_cli.py"),
        "--model_cfg",
        str(MODEL_CFG),
        "--ckpt_file",
        str(CKPT_FILE),
        "--vocab_file",
        str(VOCAB_FILE),  # ✅ 正确的中文vocab
        "--ref_audio",
        str(REF_AUDIO),
        "--ref_text",
        REF_TEXT,
        "--gen_text",
        text,
        "--output_dir",
        str(OUTPUT_DIR),
        "--output_file",
        output_name,
        "--nfe_step",
        str(NFE_STEP),
        "--cfg_strength",
        str(CFG_STRENGTH),
    ]

    if REMOVE_SILENCE:
        cmd.append("--remove_silence")

    # 设置CUDA设备
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print("开始生成...")
    print("（这可能需要几分钟时间，请耐心等待）")
    print()

    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(ROOT_DIR),
            timeout=600,  # 10分钟超时
        )

        if result.returncode != 0:
            print("❌ 生成失败！")
            print("错误信息:")
            error_msg = result.stderr if result.stderr else result.stdout
            print(error_msg[:500])
            return None
        else:
            # 验证输出文件
            if output_path.exists() and output_path.stat().st_size > 0:
                size_kb = output_path.stat().st_size / 1024
                print("✅ 生成成功！")
                print(f"文件路径: {output_path}")
                print(f"文件大小: {size_kb:.1f} KB")
                print()
                print("=" * 60)
                print("您可以:")
                print(f"1. 直接在文件资源管理器中打开: {output_path}")
                print("2. 双击文件使用默认播放器播放")
                print("3. 使用Windows Media Player播放")
                print("=" * 60)
                return output_path
            else:
                print("❌ 输出文件不存在或为空")
                return None

    except subprocess.TimeoutExpired:
        print("❌ 生成超时（超过10分钟）")
        return None
    except Exception as e:
        print(f"❌ 生成异常: {str(e)}")
        return None


def main():
    """主函数"""
    # 获取要生成的文本
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        # 默认文本（测试用）
        text = "今天天气很好，我们去公园玩吧！"
        print("未指定文本，使用默认文本:")
        print(f"  {text}")
        print()

    # 生成语音
    output_path = generate_speech(text)

    if output_path:
        print(f"\n🎉 语音已生成: {output_path}")
    else:
        print("\n❌ 语音生成失败，请检查错误信息")


if __name__ == "__main__":
    main()
