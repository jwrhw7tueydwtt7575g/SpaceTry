import pytest

try:
    import optuna  # noqa: F401
except ImportError:
    pytest.skip("optuna unavailable; expensive studies are never claimed", allow_module_level=True)
else:
    pytest.skip("expensive HPO studies require explicit opt-in; bounded orchestration is covered in test_hpo.py",
                allow_module_level=True)
