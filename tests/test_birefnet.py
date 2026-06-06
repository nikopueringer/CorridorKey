"""Unit tests for BiRefNetModule.wrapper — no network, GPU, or weights required.

The model load (snapshot_download + AutoModelForImageSegmentation.from_pretrained)
is mocked, so these exercise only the handler's load contract and dtype logic.
"""

from unittest import mock

import pytest

from BiRefNetModule.wrapper import BiRefNetHandler


def _make_handler(device):
    """Build a BiRefNetHandler with the heavy externals mocked out.

    Returns (handler, model_mock, from_pretrained_mock).
    """
    with (
        mock.patch("BiRefNetModule.wrapper.snapshot_download"),
        mock.patch("BiRefNetModule.wrapper.AutoModelForImageSegmentation") as mock_cls,
    ):
        model = mock.MagicMock()
        mock_cls.from_pretrained.return_value = model
        handler = BiRefNetHandler(device=device, usage="General")
    return handler, model, mock_cls.from_pretrained


def test_loads_with_trust_remote_code():
    """BiRefNet ships a custom architecture in its HF repo, so the weights cannot
    load without executing that code. Regression for issue #230: with
    trust_remote_code=False, from_pretrained raises "contains custom code which
    must be executed"."""
    _, _, from_pretrained = _make_handler("cpu")
    from_pretrained.assert_called_once()
    _, kwargs = from_pretrained.call_args
    assert kwargs.get("trust_remote_code") is True


@pytest.mark.parametrize(
    "device, expect_half",
    [("cpu", False), ("mps", False), ("cuda", True), ("cuda:0", True)],
)
def test_half_precision_only_on_cuda(device, expect_half):
    """fp16 is unstable on Apple's MPS backend (BiRefNet's swin attention emits
    NaNs), so half precision must be applied only on CUDA. The flag is stored on
    the instance so the model weights and the input tensor (in process()) always
    agree on dtype."""
    handler, model, _ = _make_handler(device)
    assert handler.use_half is expect_half
    assert model.half.called is expect_half
