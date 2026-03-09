"""
train_distill.py — knowledge distillation training script for GreenFormerSmall.

Trains a lightweight GreenFormerSmall student model by distilling knowledge
from a pre-trained GreenFormer teacher using the torchdistill framework.

Usage
-----
From the repository root::

    python -m distillation.train_distill \\
        --config  distillation/configs/distill_greenformer.yaml \\
        --data-root /path/to/dataset \\
        --teacher-ckpt /path/to/teacher.pth \\
        --output-dir ./checkpoints/student \\
        [--img-size 512] \\
        [--device cpu|cuda|cuda:0] \\
        [--log-freq 10]

Arguments
---------
--config
    Path to the torchdistill YAML config file.
--data-root
    Root directory containing ``train/`` and ``val/`` sub-directories.
    Each split must have ``input/``, ``alpha_hint/``, and ``alpha_gt/``
    sub-directories (``fg_gt/`` is optional).
--teacher-ckpt
    Path to the teacher GreenFormer checkpoint (``.pth`` state-dict file).
--output-dir
    Directory where the trained student checkpoint will be written.
    The final checkpoint is saved as ``student.pth``.
--img-size
    Square resolution used for training (default: 512).
--device
    PyTorch device string (default: ``cuda`` if available, else ``cpu``).
--log-freq
    How often (in batches) to print the running loss (default: 50).
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
from torchdistill.common.yaml_util import load_yaml_file
from torchdistill.core.distillation import DistillationBox

# Register custom losses BEFORE building the DistillationBox so that
# torchdistill's registry can find them by name from the YAML config.
import distillation.losses  # noqa: F401 — side-effect: registers MatteTaskLoss / MatteResponseKDLoss
from CorridorKeyModule.core.model_transformer import GreenFormer, GreenFormerSmall
from distillation.dataset import GreenScreenMattingDataset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Distil GreenFormer → GreenFormerSmall via torchdistill")
    p.add_argument("--config", required=True, help="Path to torchdistill YAML config file")
    p.add_argument("--data-root", required=True, help="Root dataset directory (contains train/ and val/)")
    p.add_argument("--teacher-ckpt", required=True, help="Path to teacher GreenFormer checkpoint (.pth)")
    p.add_argument("--output-dir", required=True, help="Directory to save the trained student checkpoint")
    p.add_argument("--img-size", type=int, default=512, help="Training image resolution (default: 512)")
    p.add_argument(
        "--device",
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="PyTorch device (default: cuda if available)",
    )
    p.add_argument("--log-freq", type=int, default=50, help="Log loss every N batches (default: 50)")
    return p


def _load_teacher(ckpt_path: str, device: torch.device) -> GreenFormer:
    """Instantiate GreenFormer and load weights from *ckpt_path*."""
    teacher = GreenFormer()
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Support both raw state-dicts and dicts with a 'model' key
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    teacher.load_state_dict(state, strict=False)
    teacher.eval()
    return teacher


def _save_checkpoint(model: torch.nn.Module, output_dir: str, filename: str = "student.pth") -> None:
    os.makedirs(output_dir, exist_ok=True)
    save_path = Path(output_dir) / filename
    state = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
    torch.save(state, save_path)
    print(f"Saved student checkpoint → {save_path}")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    data_root = Path(args.data_root)

    # ------------------------------------------------------------------
    # 1.  Datasets
    # ------------------------------------------------------------------
    train_dataset = GreenScreenMattingDataset(data_root / "train", img_size=args.img_size)
    val_dataset = GreenScreenMattingDataset(data_root / "val", img_size=args.img_size)
    dataset_dict = {"train": train_dataset, "val": val_dataset}

    print(f"Train samples : {len(train_dataset)}")
    print(f"Val   samples : {len(val_dataset)}")

    # ------------------------------------------------------------------
    # 2.  Models
    # ------------------------------------------------------------------
    teacher = _load_teacher(args.teacher_ckpt, device).to(device)
    student = GreenFormerSmall(img_size=args.img_size).to(device)

    # ------------------------------------------------------------------
    # 3.  torchdistill config
    # ------------------------------------------------------------------
    train_config: dict = load_yaml_file(args.config)

    # Inject checkpoint paths from CLI so the YAML stays path-agnostic.
    train_config.setdefault("teacher", {})["src_ckpt"] = args.teacher_ckpt
    train_config.setdefault("student", {})["dst_ckpt"] = str(Path(args.output_dir) / "student.pth")

    # ------------------------------------------------------------------
    # 4.  DistillationBox
    # ------------------------------------------------------------------
    distillation_box = DistillationBox(
        teacher_model=teacher,
        student_model=student,
        dataset_dict=dataset_dict,
        train_config=train_config,
        device=device,
        device_ids=None,
        distributed=False,
        lr_factor=1.0,
    )

    # ------------------------------------------------------------------
    # 5.  Training loop
    # ------------------------------------------------------------------
    num_epochs = distillation_box.num_epochs
    print(f"Starting distillation for {num_epochs} epochs on {device}")

    for epoch in range(num_epochs):
        distillation_box.pre_epoch_process(epoch=epoch)

        running_loss = 0.0
        t0 = time.time()
        for batch_idx, (sample_batch, targets, supp_dict) in enumerate(distillation_box.train_data_loader):
            sample_batch = sample_batch.to(device)
            if isinstance(targets, dict):
                targets = {k: v.to(device) for k, v in targets.items()}
            else:
                targets = targets.to(device)

            distillation_box.pre_forward_process(sample_batch=sample_batch, targets=targets, supp_dict=supp_dict)
            batch_loss = distillation_box.forward_process(sample_batch, targets, supp_dict)
            distillation_box.post_forward_process(loss=batch_loss)

            running_loss += batch_loss.item()
            if (batch_idx + 1) % args.log_freq == 0:
                avg = running_loss / args.log_freq
                elapsed = time.time() - t0
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}]  "
                    f"Step [{batch_idx + 1}/{len(distillation_box.train_data_loader)}]  "
                    f"Loss: {avg:.6f}  ({elapsed:.1f}s)"
                )
                running_loss = 0.0
                t0 = time.time()

        distillation_box.post_epoch_process()

    # ------------------------------------------------------------------
    # 6.  Save final student checkpoint
    # ------------------------------------------------------------------
    _save_checkpoint(distillation_box.student_model, args.output_dir)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = _build_arg_parser()
    train(parser.parse_args())
