"""Smoke tests: verify all CorridorKey packages import without error.

These catch missing __init__.py files, broken relative imports, and
missing dependencies before any real logic runs.
"""


def test_import_corridorkey_module():
    import CorridorKeyModule  # noqa: F401


def test_import_color_utils():
    from CorridorKeyModule.core import color_utils  # noqa: F401


def test_import_inference_engine():
    from CorridorKeyModule import inference_engine  # noqa: F401


def test_import_model_transformer():
    from CorridorKeyModule.core import model_transformer  # noqa: F401


def test_import_greenformersmall():
    from CorridorKeyModule.core.model_transformer import GreenFormerSmall  # noqa: F401


def test_import_distillation():
    import distillation  # noqa: F401


def test_import_distillation_dataset():
    from distillation import dataset  # noqa: F401


def test_import_distillation_losses():
    from distillation import losses  # noqa: F401


def test_import_gvm_core():
    import gvm_core  # noqa: F401


def test_import_gvm_wrapper():
    from gvm_core import wrapper  # noqa: F401


def test_import_videomama():
    import VideoMaMaInferenceModule  # noqa: F401


def test_import_videomama_inference():
    from VideoMaMaInferenceModule import inference  # noqa: F401
