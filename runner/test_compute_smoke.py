import importlib.util
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("compute_smoke.py")
spec = importlib.util.spec_from_file_location("compute_smoke", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_choose_ram_bytes_caps_at_maximum():
    assert module.choose_ram_bytes(64 * module.GIB, 8.0) == 8 * module.GIB


def test_choose_ram_bytes_uses_quarter_of_available_memory():
    assert module.choose_ram_bytes(8 * module.GIB, 8.0) == 2 * module.GIB


def test_choose_ram_bytes_handles_invalid_values():
    assert module.choose_ram_bytes(0, 8.0) == 0
    assert module.choose_ram_bytes(8 * module.GIB, 0) == 0


def test_matrix_size_is_bounded():
    assert module.choose_matrix_size(0) == 2048
    assert 2048 <= module.choose_matrix_size(8 * module.GIB) <= 8192
    assert module.choose_matrix_size(10**15) == 8192
