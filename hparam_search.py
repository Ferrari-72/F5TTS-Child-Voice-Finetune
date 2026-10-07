"""
推理参数优化脚本 - 测试不同参数组合以找到最佳配置
Inference Parameter Optimization Script - Test different parameter combinations
"""
import os
import sys
import subprocess
import json
from pathlib import Path
from itertools import product

ROOT_DIR = Path(__file__).parent.absolute()
os.chdir(ROOT_DIR)

# 测试参数范围
NFE_STEPS = [50, 75, 100, 125]  # 采样步数
CFG_STRENGTHS = [2.0, 2.5, 3.0]  # 引导强度

# 测试文件（使用少量文件进行快速测试）
TEST_FILE = ROOT_DIR / "data" / "test_data" / "YZ-0002.wav"
REF_AUDIO = ROOT_DIR / "data" / "child-tts" / "000018.wav"
REF_TEXT = "城里有好多游乐场，可好玩儿了！"

# 模型路径
MODEL_CFG = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"
CKPT_FILE = ROOT_DIR / "ckpts" / "child-tts_pinyin" / "model_last.pt"
VOCAB_FILE = ROOT_DIR / "ckpts" / "child-tts" / "vocab.txt"  # pretrained pinyin vocab (2545 tokens); the dataset-generated char vocab breaks pinyin tokenization

OUTPUT_DIR = ROOT_DIR / "outputs" / "param_optimization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def run_inference(nfe_step, cfg_strength, output_name):
    """运行推理"""
    import whisper
    
    # 转录测试音频
    model = whisper.load_model("large-v2")
    result = model.transcribe(str(TEST_FILE), language="zh")
    gen_text = result["text"].strip()
    
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
        "--nfe_step", str(nfe_step),
        "--cfg_strength", str(cfg_strength),
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
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        return False

def evaluate_audio(generated_audio):
    """评估音频质量（快速评估，使用非侵入式指标）"""
    try:
        sys.path.insert(0, str(ROOT_DIR))
        from speechscore import SpeechScore
        
        evaluator = SpeechScore(['DNSMOS', 'SRMR'])  # 非侵入式指标，不需要参考音频
        
        scores = evaluator(
            test_path=str(generated_audio),
            reference_path=None,
            window=None,
            score_rate=16000,
            return_mean=False
        )
        
        # 提取分数
        dnsmos = scores.get('DNSMOS', {}).get('mos', 0) if isinstance(scores.get('DNSMOS'), dict) else 0
        srmr = scores.get('SRMR', 0) if isinstance(scores.get('SRMR'), (int, float)) else 0
        
        return {
            'DNSMOS': dnsmos,
            'SRMR': srmr,
            'combined_score': dnsmos * 0.6 + srmr * 0.4  # 加权组合
        }
    except Exception as e:
        print(f"⚠️ 评估失败: {str(e)}")
        return None

def main():
    """主函数：参数优化"""
    print("=" * 60)
    print("推理参数优化")
    print("=" * 60)
    
    results = []
    
    # 测试所有参数组合
    total_combinations = len(NFE_STEPS) * len(CFG_STRENGTHS)
    current = 0
    
    for nfe_step, cfg_strength in product(NFE_STEPS, CFG_STRENGTHS):
        current += 1
        print(f"\n[{current}/{total_combinations}] 测试参数: nfe_step={nfe_step}, cfg_strength={cfg_strength}")
        
        output_name = f"optimize_nfe{nfe_step}_cfg{cfg_strength:.1f}.wav"
        
        # 运行推理
        if run_inference(nfe_step, cfg_strength, output_name):
            generated_audio = OUTPUT_DIR / output_name
            if generated_audio.exists():
                # 评估
                scores = evaluate_audio(generated_audio)
                if scores:
                    results.append({
                        'nfe_step': nfe_step,
                        'cfg_strength': cfg_strength,
                        'scores': scores,
                        'output_file': str(generated_audio)
                    })
                    print(f"  DNSMOS: {scores['DNSMOS']:.3f}, SRMR: {scores['SRMR']:.3f}, Combined: {scores['combined_score']:.3f}")
        else:
            print(f"  ❌ 推理失败")
    
    # 保存结果
    results_file = OUTPUT_DIR / "optimization_results.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 找到最佳参数
    if results:
        best = max(results, key=lambda x: x['scores']['combined_score'])
        print("\n" + "=" * 60)
        print("最佳参数组合:")
        print(f"  nfe_step: {best['nfe_step']}")
        print(f"  cfg_strength: {best['cfg_strength']}")
        print(f"  DNSMOS: {best['scores']['DNSMOS']:.3f}")
        print(f"  SRMR: {best['scores']['SRMR']:.3f}")
        print(f"  Combined Score: {best['scores']['combined_score']:.3f}")
        print("=" * 60)
    
    print(f"\n结果已保存: {results_file}")

if __name__ == "__main__":
    main()

