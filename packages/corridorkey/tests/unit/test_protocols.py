"""Unit tests for the AlphaGenerator protocol."""

from __future__ import annotations

from corridorkey.protocols import AlphaGenerator


class _ConcreteGenerator:
    """Minimal valid implementation of AlphaGenerator."""

    @property
    def name(self) -> str:
        return "test_gen"

    def generate(self, clip, on_progress=None, on_warning=None) -> None:
        pass


class _MissingName:
    def generate(self, clip, on_progress=None, on_warning=None) -> None:
        pass


class _MissingGenerate:
    @property
    def name(self) -> str:
        return "x"


class TestAlphaGeneratorProtocol:
    def test_valid_implementation_is_instance(self):
        gen = _ConcreteGenerator()
        assert isinstance(gen, AlphaGenerator)

    def test_missing_name_not_instance(self):
        assert not isinstance(_MissingName(), AlphaGenerator)

    def test_missing_generate_not_instance(self):
        assert not isinstance(_MissingGenerate(), AlphaGenerator)

    def test_name_returns_string(self):
        gen = _ConcreteGenerator()
        assert isinstance(gen.name, str)
        assert gen.name == "test_gen"

    def test_generate_callable(self):
        gen = _ConcreteGenerator()
        # Should not raise
        gen.generate(None)
