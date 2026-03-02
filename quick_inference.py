"""
快速测试脚本 - 测试优化后的推理参数
Quick Test Script - Test optimized inference parameters
"""
import os
import sys
import subprocess
from pathlib import Path
import whisper

ROOT_DIR = Path(__file__).parent.absolute()
os.chdir(ROOT_DIR)

# 测试文件
TEST_FILE = ROOT_DIR / "data" / "test_data" / "YZ-0002.wav"
REF_AUDIO = ROOT_DIR / "data" / "child-tts" / "000018.wav"
REF_TEXT = "城里有好多游乐场，可好玩儿了！"

# 模型路径
MODEL_CFG = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"
CKPT_FILE = ROOT_DIR / "ckpts" / "child-tts_processed" / "model_last.pt"
VOCAB_FILE = ROOT_DIR / "data" / "child-tts_processed" / "vocab.txt"  # 使用中文vocab，不是预训练的英文vocab

OUTPUT_DIR = ROOT_DIR / "outputs" / "quick_test"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 优化的参数
NFE_STEP = 150
CFG_STRENGTH = 3.0

def main():
    print("=" * 60)
    print("快速测试 - 优化后的推理参数")
    print("=" * 60)
    print(f"nfe_step: {NFE_STEP}")
    print(f"cfg_strength: {CFG_STRENGTH}")
    print("-" * 60)
    
    # 转录测试音频
    print("转录测试音频...")
    model = whisper.load_model("large-v2")
    result = model.transcribe(str(TEST_FILE), language="zh")
    gen_text = result["text"].strip()
    print(f"转录文本: {gen_text}")
    
    # 运行推理
    print("\n运行推理...")
    output_name = "quick_test_optimized.wav"
    output_path = OUTPUT_DIR / output_name
    
    cmd = [
        sys.executable,
        str(ROOT_DIR / "src" / "f5_tts" / "infer" / "infer_cli.py"),
        "--model_cfg", str(MODEL_CFG),
        "--ckpt_file", str(CKPT_FILE),
        "--vocab_file", str(VOCAB_FILE),
        "--ref_audio", str(REF_AUDIO),
        "--ref_text", REF_TEXT,
        "--gen_text", gen_text,
        "--output_dir", str(OUTPUT_DIR),
        "--output_file", output_name,
        "--nfe_step", str(NFE_STEP),
        "--cfg_strength", str(CFG_STRENGTH),
    ]
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(ROOT_DIR)
        )
        
        if result.returncode == 0:
            print(f"✅ 推理成功！输出文件: {output_path}")
            if output_path.exists():
                size_kb = output_path.stat().st_size / 1024
                print(f"   文件大小: {size_kb:.1f} KB")
        else:
            print(f"❌ 推理失败:")
            print(result.stderr[:500])
    except Exception as e:
        print(f"❌ 推理异常: {str(e)}")

if __name__ == "__main__":
    main()

