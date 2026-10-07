import argparse
import codecs
import os
import re
from datetime import datetime
from importlib.resources import files
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repo root
import numpy as np
import soundfile as sf
import tomli
from cached_path import cached_path
from hydra.utils import get_class
from omegaconf import OmegaConf
from src.f5_tts.Models import DiT,UNetT,CFM
from src.f5_tts.infer.utils_infer import (
    mel_spec_type,
    target_rms,
    cross_fade_duration,
    nfe_step,
    cfg_strength,
    sway_sampling_coef,
    speed,
    fix_duration,
    device,
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
    remove_silence_for_generated_wav,
)


parser = argparse.ArgumentParser(
    prog="python3 infer-cli.py",
    description="Commandline interface for E2/F5 TTS with Advanced Batch Processing.",
    epilog="Specify options above to override one or more settings from config.",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default='./src/f5_tts/infer/examples/basic/basic.toml',
    help="The configuration file, default see infer/examples/basic/basic.toml",
)


# Note. Not to provide default value here in order to read default from config file

parser.add_argument(
    "-m",
    "--model",
    type=str,
    help="The model name: F5TTS_v1_Base | F5TTS_Base | E2TTS_Base | etc.",
)
parser.add_argument(
    "-mc",
    "--model_cfg",
    type=str,
    help="The path to F5-TTS model config file .yaml",
)
parser.add_argument(
    "-p",
    "--ckpt_file",
    type=str,
    help="The path to model checkpoint .pt, leave blank to use default",
)
parser.add_argument(
    "-v",
    "--vocab_file",
    type=str,
    help="The path to vocab file .txt, leave blank to use default",
)
parser.add_argument(
    "-r",
    "--ref_audio",
    type=str,
    help="The reference audio file.",
)
parser.add_argument(
    "-s",
    "--ref_text",
    type=str,
    help="The transcript/subtitle for the reference audio",
)
parser.add_argument(
    "-t",
    "--gen_text",
    type=str,
    help="The text to make model synthesize a speech",
)
parser.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="The path to output folder",
)
parser.add_argument(
    "-w",
    "--output_file",
    type=str,
    help="The name of output file",
)
parser.add_argument(
    "--save_chunk",
    action="store_true",
    help="To save each audio chunks during inference",
)
parser.add_argument(
    "--remove_silence",
    action="store_true",
    help="To remove long silence found in ouput",
)
parser.add_argument(
    "--load_vocoder_from_local",
    action="store_true",
    # default = ''
    help="To load vocoder from local dir, default to ../checkpoints/vocos-mel-24khz",
)
parser.add_argument(
    "--vocoder_name",
    type=str,
    choices=["vocos", "bigvgan"],
    help=f"Used vocoder name: vocos | bigvgan, default {mel_spec_type}",
)
parser.add_argument(
    "--target_rms",
    type=float,
    help=f"Target output speech loudness normalization value, default {target_rms}",
)
parser.add_argument(
    "--cross_fade_duration",
    type=float,
    help=f"Duration of cross-fade between audio segments in seconds, default {cross_fade_duration}",
)
parser.add_argument(
    "--nfe_step",
    type=int,
    help=f"The number of function evaluation (denoising steps), default {nfe_step}",
)
parser.add_argument(
    "--cfg_strength",
    type=float,
    help=f"Classifier-free guidance strength, default {cfg_strength}",
)
parser.add_argument(
    "--sway_sampling_coef",
    type=float,
    help=f"Sway Sampling coefficient, default {sway_sampling_coef}",
)
parser.add_argument(
    "--speed",
    type=float,
    help=f"The speed of the generated audio, default {speed}",
)
parser.add_argument(
    "--fix_duration",
    type=float,
    help=f"Fix the total duration (ref and gen audios) in seconds, default {fix_duration}",
)
parser.add_argument(
    "--device",
    type=str,
    help="Specify the device to run on",
)
args = parser.parse_args()


# config file

config = tomli.load(open(args.config, "rb"))


# command-line interface parametter
ref_audio = args.ref_audio or config.get("ref_audio","")
ref_text = args.ref_text or config.get("ref_text","")
gen_text = args.gen_text or config.get("gen_text","")
model = args.model or config.get("model", "F5TTS_v1_Base")
ckpt_file = args.ckpt_file or config.get("ckpt_file", "")
vocab_file = args.vocab_file or config.get("vocab_file", "")

output_dir = args.output_dir or config.get("output_dir", "") or "test_outputs"
output_file = args.output_file or config.get("output_file", "") or "infer_output.wav"

load_vocoder_from_local = args.load_vocoder_from_local or config.get("load_vocoder_from_local", False)

vocoder_name = args.vocoder_name or config.get("vocoder_name", mel_spec_type)
target_rms = args.target_rms or config.get("target_rms", target_rms)
cross_fade_duration = args.cross_fade_duration or config.get("cross_fade_duration", cross_fade_duration)
nfe_step = args.nfe_step or config.get("nfe_step", nfe_step)
cfg_strength = args.cfg_strength or config.get("cfg_strength", cfg_strength)
sway_sampling_coef = args.sway_sampling_coef or config.get("sway_sampling_coef", sway_sampling_coef)
speed = args.speed or config.get("speed", speed)
fix_duration = args.fix_duration or config.get("fix_duration", fix_duration)
device = args.device or config.get("device", device)
save_chunk = args.save_chunk or config.get("save_chunk",False)

os.makedirs(output_dir,exist_ok=True)
wave_path = Path(output_dir) / output_file
# spectrogram_path = Path(output_dir) / "infer_cli_out.png"
if save_chunk:
    output_chunk_dir = os.path.join(output_dir, f"{Path(output_file).stem}_chunks")
    if not os.path.exists(output_chunk_dir):
        os.makedirs(output_chunk_dir)


# load vocoder

if vocoder_name == "vocos":
    vocoder_local_path = "../checkpoints/vocos-mel-24khz"
elif vocoder_name == "bigvgan":
    vocoder_local_path = "../checkpoints/bigvgan_v2_24khz_100band_256x"

vocoder = load_vocoder(
    vocoder_name=vocoder_name, is_local=load_vocoder_from_local, local_path=vocoder_local_path, device=device
)


# load TTS model

_model_cfg_path = args.model_cfg or config.get("model_cfg", "")
if not _model_cfg_path:
    # 默认使用 F5TTS_v1_Base 配置
    _model_cfg_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../configs/F5TTS_v1_Base.yaml"
    )
model_cfg = OmegaConf.load(_model_cfg_path)
if model_cfg.model.backbone == "DiT":
    model_cls = DiT
    model_arch = model_cfg.model.arch
elif model_cfg.model.backbone == "UNetT":
    model_cls = UNetT
    model_arch = model_cfg.model.arch
else:
    raise ValueError(f"Unknown backbone: {model_cfg.model.backbone}")

if model != "F5TTS_Base":
    assert vocoder_name == model_cfg.model.mel_spec.mel_spec_type, \
        f"Vocoder mismatch: {vocoder_name} vs config {model_cfg.model.mel_spec.mel_spec_type}"

print(f"Using {model}...")
ema_model = load_model(
    model_cls, model_arch, ckpt_file, mel_spec_type=vocoder_name, vocab_file=vocab_file, device=device
)


# inference process


def main():

        audio_segment, final_sample_rate, spectragram = infer_process(
            ref_audio,
            ref_text,
            gen_text,
            ema_model,
            vocoder,
            mel_spec_type=vocoder_name,
            target_rms=target_rms,
            cross_fade_duration=cross_fade_duration,
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            fix_duration=fix_duration,
            device=device)
        
        with open(wave_path, "wb") as f:
            sf.write(f.name, audio_segment, final_sample_rate)


if __name__ == "__main__":
    main()
