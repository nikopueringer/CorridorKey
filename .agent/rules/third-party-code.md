# Third-Party Code

The `gvm_core/` and `VideoMaMaInferenceModule/` directories contain upstream-derived research code. Do NOT:
- Reformat, relint, or restructure
- Add type annotations
- Change import style or naming conventions
- Apply ruff fixes (these dirs are excluded from ruff config)

Acceptable modifications:
- Guarding platform-specific calls (e.g., `if torch.cuda.is_available(): torch.cuda.empty_cache()`)
- Fixing crashes that block the main pipeline
- Minimal changes to default parameter values (e.g., `device="cuda"` to `device="cpu"`)

Note: `gvm_core/wrapper.py` and `VideoMaMaInferenceModule/inference.py` are project-authored facades wrapping upstream code. These CAN be modified with normal code quality standards, but keep changes minimal to avoid diverging from upstream patterns.

**Licenses:**
- `gvm_core/`: CC BY-NC-SA 4.0 ([aim-uofa/GVM](https://github.com/aim-uofa/GVM))
- `VideoMaMaInferenceModule/`: CC BY-NC 4.0 + Stability AI Community License ([cvlab-kaist/VideoMaMa](https://github.com/cvlab-kaist/VideoMaMa))
