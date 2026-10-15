from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_phase1_layout_is_complete():
    assert (ROOT / "package.xml").is_file()
    assert (ROOT / "CMakeLists.txt").is_file()
    assert (ROOT / "worlds" / "leo_orbit.sdf").is_file()
    assert (ROOT / "models" / "chaser" / "model.sdf").is_file()
    assert (ROOT / "models" / "target" / "model.sdf").is_file()
    assert (ROOT / "src" / "cw_dynamics_system.cpp").is_file()
    text = (ROOT / "INTERFACES.md").read_text()
    for token in ("LVLH", "m/s^2", "command/accel", "reference oracle"):
        assert token in text
