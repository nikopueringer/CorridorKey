# Third-Party Code — gvm_core

This directory contains upstream-derived research code from [aim-uofa/GVM](https://github.com/aim-uofa/GVM) (CC BY-NC-SA 4.0). Do NOT:
- Reformat, relint, or restructure
- Add type annotations
- Change import style or naming conventions
- Apply ruff fixes (this dir is excluded from ruff config)

Acceptable modifications:
- Guarding platform-specific calls (e.g., `if torch.cuda.is_available(): torch.cuda.empty_cache()`)
- Fixing crashes that block the main pipeline
- Minimal changes to default parameter values (e.g., `device="cuda"` to `device="cpu"`)

Note: `wrapper.py` is a project-authored facade wrapping upstream code. It CAN be modified with normal code quality standards, but keep changes minimal to avoid diverging from upstream patterns.
