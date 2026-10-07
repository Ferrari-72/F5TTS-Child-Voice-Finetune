"""Gradio web demo for the fine-tuned child-voice F5-TTS model.

Usage (from repo root):
    python app.py [--ckpt ckpts/child-tts_pinyin/model_last.pt] [--share]
"""

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

import gradio as gr
import numpy as np
import soundfile as sf
from omegaconf import OmegaConf

from src.f5_tts.Models import DiT, UNetT
from src.f5_tts.infer.utils_infer import infer_process, load_model, load_vocoder

DEFAULT_CKPT = ROOT_DIR / "ckpts" / "child-tts_pinyin" / "model_last.pt"
VOCAB_FILE = ROOT_DIR / "ckpts" / "child-tts" / "vocab.txt"  # pretrained pinyin vocab
MODEL_CFG = ROOT_DIR / "src" / "f5_tts" / "configs" / "F5TTS_v1_Base.yaml"
DEFAULT_REF_AUDIO = ROOT_DIR / "data" / "child-tts" / "000018.wav"
DEFAULT_REF_TEXT = "城里有好多游乐场，可好玩儿了！"

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", default=str(DEFAULT_CKPT))
parser.add_argument("--share", action="store_true")
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()

print("Loading vocoder and model...")
vocoder = load_vocoder(vocoder_name="vocos", is_local=False, local_path=None)
model_cfg = OmegaConf.load(str(MODEL_CFG))
model_cls = DiT if model_cfg.model.backbone == "DiT" else UNetT
ema_model = load_model(
    model_cls, model_cfg.model.arch, str(args.ckpt), mel_spec_type="vocos", vocab_file=str(VOCAB_FILE)
)
print("Model loaded.")


def synthesize(ref_audio, ref_text, gen_text, nfe_step, cfg_strength, speed):
    if ref_audio is None:
        raise gr.Error("请提供参考音频 / Please provide a reference audio.")
    if not gen_text.strip():
        raise gr.Error("请输入要合成的文本 / Please enter text to synthesize.")

    audio_segment, sample_rate, _ = infer_process(
        ref_audio,
        ref_text.strip(),
        gen_text.strip(),
        ema_model,
        vocoder,
        mel_spec_type="vocos",
        nfe_step=int(nfe_step),
        cfg_strength=cfg_strength,
        speed=speed,
    )
    return sample_rate, np.array(audio_segment)


with gr.Blocks(title="F5-TTS Child Voice") as demo:
    gr.Markdown("# F5-TTS Child Voice Cloning\nFine-tuned F5-TTS v1 Base for Mandarin child voice synthesis.")

    with gr.Row():
        with gr.Column():
            ref_audio = gr.Audio(
                label="Reference Audio / 参考音频", type="filepath", value=str(DEFAULT_REF_AUDIO)
            )
            ref_text = gr.Textbox(label="Reference Text / 参考音频文本", value=DEFAULT_REF_TEXT)
            gen_text = gr.Textbox(
                label="Text to Synthesize / 合成文本",
                value="今天天气很好，我们去公园玩吧！",
                lines=3,
            )
            with gr.Accordion("Advanced / 推理参数", open=False):
                nfe_step = gr.Slider(16, 128, value=50, step=1, label="nfe_step（采样步数，越高越好但越慢）")
                cfg_strength = gr.Slider(1.0, 5.0, value=2.0, step=0.1, label="cfg_strength（参考忠实度）")
                speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="speed（语速）")
            btn = gr.Button("Synthesize / 合成", variant="primary")

        with gr.Column():
            out_audio = gr.Audio(label="Generated / 生成结果", type="numpy")

    btn.click(
        synthesize,
        inputs=[ref_audio, ref_text, gen_text, nfe_step, cfg_strength, speed],
        outputs=out_audio,
    )

if __name__ == "__main__":
    demo.launch(server_port=args.port, share=args.share)
