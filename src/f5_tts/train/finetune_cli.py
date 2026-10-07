import argparse
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))  # repo root
import shutil
from importlib.resources import files

from cached_path import cached_path

from src.f5_tts.Models import CFM, UNetT, DiT, Trainer
from src.f5_tts.Models.utils import get_tokenizer
from src.f5_tts.Models.Datasets import load_dataset, collate_fn
from torch.utils.tensorboard import SummaryWriter
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
import math


# -------------------------- Dataset Settings --------------------------- #
target_sample_rate = 24000
n_mel_channels = 100
hop_length = 256
win_length = 1024
n_fft = 1024
mel_spec_type = "vocos"  


# -------------------------- Argument Parsing --------------------------- #
def parse_args():
    parser = argparse.ArgumentParser(description="Train CFM Model")

    parser.add_argument(
        "--exp_name",
        type=str,
        default="F5TTS_v1_Base",
        choices=["F5TTS_v1_Base", "F5TTS_Base", "E2TTS_Base"],
        help="Experiment name",
    )
    parser.add_argument("--dataset_name", type=str, default="child-tts_pinyin", help="Name of the dataset folder under data/ (text must be pinyin-tokenized)")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate for training")
    parser.add_argument("--batch_size_per_gpu", type=int, default=4, help="Batch size per GPU (samples)")
    parser.add_argument(
        "--batch_size_type", type=str, default="sample", choices=["frame", "sample"], help="Batch size type"
    )
    parser.add_argument("--max_samples", type=int, default=4, help="Max sequences per batch")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    # 修复关键Bug: warmup_updates 由 -1(自动) 表示 10% 总步数, 原20000远超小数据集总步数
    parser.add_argument("--num_warmup_updates", type=int, default=-1,
                        help="Warmup steps. -1 = auto (10%% of total steps). For child-tts ~220 samples, "
                             "batch=2, 50 epochs => ~5500 total steps, so ~550 warmup. "
                             "Old default 20000 was larger than total steps, causing near-zero LR throughout training!")
    parser.add_argument("--save_per_updates", type=int, default=300, help="Save checkpoint every N updates")
    parser.add_argument(
        "--keep_last_n_checkpoints",
        type=int,
        default=-1,
        help="-1 to keep all, 0 to not save intermediate, > 0 to keep last N checkpoints",
    )
    parser.add_argument("--last_per_updates", type=int, default=500, help="Save last checkpoint every N updates")
    parser.add_argument("--finetune", type=bool, default=True, help="Use Finetune (default: True)")

    parser.add_argument(
        "--pretrain", type=str,
        default='ckpts/child-tts/pretrained_model_1250000.safetensors',
        help="the path to pretrained checkpoint"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="custom", choices=["pinyin", "char", "custom"], help="Tokenizer type"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default='ckpts/child-tts/vocab.txt',
        help="Path to custom tokenizer vocab file (only used if tokenizer = 'custom'). "
             "Must be the PRETRAINED pinyin vocab (2545 tokens) when fine-tuning from F5TTS_v1_Base.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log inferenced samples per ckpt save updates",
    )
    parser.add_argument("--logger", type=str, default="tensorboard", choices=[None, "wandb", "tensorboard"], help="logger")
    parser.add_argument(
        "--bnb_optimizer",
        action="store_true",
        help="Use 8-bit Adam optimizer from bitsandbytes",
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["linear", "cosine"],
        help="LR scheduler type. cosine gives better convergence for fine-tuning small datasets."
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no", choices=["no", "fp16", "bf16"],
        help="Mixed precision training. bf16 for RTX 30/40 series (~2x speed, half VRAM); fp16 for Kaggle P100/T4."
    )

    return parser.parse_args()


# -------------------------- 新增：自定义TensorBoard日志类（适配Trainer） --------------------------
class TensorBoardLogger:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)
        self.global_step = 0

    def log(self, metrics, step=None):
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                value = value.item()
            self.writer.add_scalar(key, value, self.global_step)

    def close(self):
        self.writer.close()


# -------------------------- Training Settings -------------------------- #
def main():
    args = parse_args()

    checkpoint_path = os.path.join(f"./ckpts/{args.dataset_name}")
    if args.finetune:
        print('------begin fineyuning your model!--------')

    if args.exp_name == "F5TTS_v1_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            conv_layers=4,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "F5TTS_Base":
        wandb_resume_id = None
        model_cls = DiT
        model_cfg = dict(
            dim=1024,
            depth=22,
            heads=16,
            ff_mult=2,
            text_dim=512,
            text_mask_padding=False,
            conv_layers=4,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/F5-TTS/F5TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    elif args.exp_name == "E2TTS_Base":
        wandb_resume_id = None
        model_cls = UNetT
        model_cfg = dict(
            dim=1024,
            depth=24,
            heads=16,
            ff_mult=4,
            text_mask_padding=False,
            pe_attn_head=1,
        )
        if args.finetune:
            if args.pretrain is None:
                ckpt_path = str(cached_path("hf://SWivid/E2-TTS/E2TTS_Base/model_1200000.pt"))
            else:
                ckpt_path = args.pretrain

    if args.finetune:
        if not os.path.isdir(checkpoint_path):
            os.makedirs(checkpoint_path, exist_ok=True)

        file_checkpoint = os.path.basename(ckpt_path)
        if not file_checkpoint.startswith("pretrained_"):
            file_checkpoint = "pretrained_" + file_checkpoint
        file_checkpoint = os.path.join(checkpoint_path, file_checkpoint)
        if not os.path.isfile(file_checkpoint):
            shutil.copy2(ckpt_path, file_checkpoint)
            print("copy checkpoint for finetune")

    tokenizer = args.tokenizer
    if tokenizer == "custom":
        if not args.tokenizer_path:
            raise ValueError("Custom tokenizer selected, but no tokenizer_path provided.")
        tokenizer_path = args.tokenizer_path
    else:
        tokenizer_path = args.dataset_name

    vocab_char_map, vocab_size = get_tokenizer(tokenizer_path, tokenizer)

    print("\nvocab : ", vocab_size)
    print("\nvocoder : ", mel_spec_type)

    mel_spec_kwargs = dict(
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mel_channels=n_mel_channels,
        target_sample_rate=target_sample_rate,
        mel_spec_type=mel_spec_type,
    )

    model = CFM(
        transformer=model_cls(**model_cfg, text_num_embeds=vocab_size, mel_dim=n_mel_channels),
        mel_spec_kwargs=mel_spec_kwargs,
        vocab_char_map=vocab_char_map,
    )

    if args.logger == "tensorboard":
        log_dir = f"runs/{args.dataset_name}_{args.exp_name}_epoch{args.epochs}_lr{args.learning_rate}"
        tensorboard_logger = TensorBoardLogger(log_dir=log_dir)
        print(f"📊 TensorBoard日志已启动，保存路径：{log_dir}")
    else:
        tensorboard_logger = None

    trainer = Trainer(
        model,
        args.epochs,
        args.learning_rate,
        num_warmup_updates=args.num_warmup_updates,
        save_per_updates=args.save_per_updates,
        keep_last_n_checkpoints=args.keep_last_n_checkpoints,
        checkpoint_path=checkpoint_path,
        batch_size_per_gpu=args.batch_size_per_gpu,
        batch_size_type=args.batch_size_type,
        max_samples=args.max_samples,
        grad_accumulation_steps=args.grad_accumulation_steps,
        max_grad_norm=args.max_grad_norm,
        logger=args.logger,
        wandb_project=args.dataset_name,
        wandb_run_name=args.exp_name,
        wandb_resume_id=wandb_resume_id,
        log_samples=args.log_samples,
        last_per_updates=args.last_per_updates,
        bnb_optimizer=args.bnb_optimizer,
        mixed_precision=args.mixed_precision,
        vocab_size=vocab_size,
        mel_spec_type=mel_spec_type
    )

    original_train = trainer.train

    def custom_train(train_dataset, resumable_with_seed=666):
        # 手动创建训练数据加载器
        train_loader = DataLoader(
            train_dataset,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True,
            batch_size=args.batch_size_per_gpu,
            shuffle=True,
            generator=torch.Generator().manual_seed(resumable_with_seed) if resumable_with_seed else None
        )

        if hasattr(trainer, 'accelerator') and trainer.accelerator is not None:
            train_loader = trainer.accelerator.prepare(train_loader)
        
        trainer.model.train()
        global_step = 0

        # ========== 关键修复：自动计算 warmup_updates ==========
        # 旧代码 default=20000 远超小数据集总训练步数，LR 从未到达目标值
        total_updates = math.ceil(len(train_loader) / trainer.grad_accumulation_steps) * trainer.epochs
        if args.num_warmup_updates == -1:
            # 自动：10% 总步数用于 warmup，但最少 50 步，最多 1000 步
            warmup_updates = max(50, min(1000, int(total_updates * 0.10)))
            print(f"📊 Auto warmup: {warmup_updates} steps (10% of {total_updates} total)")
        else:
            warmup_updates = args.num_warmup_updates
            if warmup_updates >= total_updates:
                print(f"⚠️  WARNING: num_warmup_updates ({warmup_updates}) >= total_updates ({total_updates})! "
                      f"LR will never reach target. Auto-clamping to 10% of total.")
                warmup_updates = max(50, int(total_updates * 0.10))

        decay_updates = max(1, total_updates - warmup_updates)
        print(f"📊 LR Schedule [{args.scheduler}]: warmup={warmup_updates}, decay={decay_updates}, total={total_updates}")

        # ========== 支持 cosine 和 linear 两种 LR 调度 ==========
        warmup_scheduler = LinearLR(trainer.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_updates)
        if args.scheduler == "cosine":
            # Cosine decay: 更平滑，对小数据集 fine-tuning 效果更好
            decay_scheduler = CosineAnnealingLR(trainer.optimizer, T_max=decay_updates, eta_min=1e-7)
        else:
            decay_scheduler = LinearLR(trainer.optimizer, start_factor=1.0, end_factor=1e-8, total_iters=decay_updates)

        trainer.scheduler = SequentialLR(
            trainer.optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_updates]
        )
        trainer.scheduler = trainer.accelerator.prepare(trainer.scheduler)

        start_update = trainer.load_checkpoint()
        if start_update > 0:
            global_step = start_update
            print(f"Resumed from checkpoint at update {start_update}")

        for epoch in range(trainer.epochs):
            epoch_loss = 0.0
            batch_count = 0

            for batch in train_loader:
                text_inputs = batch["text"]
                mel_spec = batch["mel"].permute(0, 2, 1)
                mel_lengths = batch.get("mel_lengths", None)

                # 使用 accelerator.autocast 混合精度加速
                with trainer.accelerator.autocast():
                    model_output = trainer.model(mel_spec, text=text_inputs, lens=mel_lengths)
                    loss = model_output[0]

                loss_val = loss.item()
                loss = loss / trainer.grad_accumulation_steps
                trainer.accelerator.backward(loss)

                if (batch_count + 1) % trainer.grad_accumulation_steps == 0:
                    if trainer.max_grad_norm > 0:
                        trainer.accelerator.clip_grad_norm_(trainer.model.parameters(), trainer.max_grad_norm)
                    trainer.optimizer.step()
                    trainer.scheduler.step()
                    trainer.optimizer.zero_grad()
                    global_step += 1

                epoch_loss += loss_val
                batch_count += 1

                if tensorboard_logger is not None:
                    tensorboard_logger.log({
                        "Train/Step Loss": loss_val,
                        "Train/Epoch": epoch,
                        "Train/Global Step": global_step,
                        "Train/Learning Rate": trainer.scheduler.get_last_lr()[0],
                    }, step=global_step)

                if global_step % trainer.save_per_updates == 0 and global_step != 0:
                    trainer.save_checkpoint(global_step)

                if global_step % trainer.last_per_updates == 0 and global_step != 0:
                    trainer.save_checkpoint(global_step, last=True)

            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
            lr_now = trainer.scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{trainer.epochs}] | Loss: {avg_epoch_loss:.4f} | LR: {lr_now:.2e} | Steps: {global_step}")

            if tensorboard_logger is not None:
                tensorboard_logger.log({
                    "Train/Epoch Average Loss": avg_epoch_loss,
                    "Train/Learning Rate": lr_now,
                }, step=global_step)

        trainer.save_checkpoint(global_step)
        trainer.save_checkpoint(global_step, last=True)

        if tensorboard_logger is not None:
            tensorboard_logger.close()
            print("✅ 训练完成，TensorBoard日志已保存")

    trainer.train = custom_train

    train_dataset = load_dataset(args.dataset_name, tokenizer, mel_spec_kwargs=mel_spec_kwargs)
    trainer.train(
        train_dataset,
        resumable_with_seed=666
    )


if __name__ == "__main__":
    main()