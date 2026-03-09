"""
CorridorKey distillation package.

Provides a torchdistill-based knowledge distillation pipeline for training
GreenFormerSmall (student) from a pre-trained GreenFormer (teacher), resulting
in a model that is ~4× smaller while retaining high matte quality.

Submodules
----------
dataset
    GreenScreenMattingDataset — loads (input, alpha_hint) frames and GT targets.
losses
    MatteTaskLoss, MatteResponseKDLoss — custom torchdistill mid-level losses.
train_distill
    CLI entry-point: ``python -m distillation.train_distill --config ...``

Example
-------
Run distillation from the repository root::

    python -m distillation.train_distill \\
        --config distillation/configs/distill_greenformer.yaml \\
        --data-root /path/to/matting_dataset \\
        --teacher-ckpt /path/to/teacher.pth \\
        --output-dir ./checkpoints/student
"""
