"""The test-only wrong-item tickets never reach training: no training code imports their
generator or opens their files."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING = [ROOT / "scripts" / "train.py",
            *[p for p in (ROOT / "thornecliffe").glob("*.py") if p.name != "test_only.py"]]


def imports(path: Path) -> set[str]:
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names |= {f"{node.module}.{a.name}" for a in node.names}
    return names


def test_training_code_never_imports_the_test_only_generator():
    for path in TRAINING:
        assert not any("test_only" in n for n in imports(path)), path


def test_the_trainer_never_opens_test_only_files():
    text = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
    assert "test_only" not in text and "test_wrong_item" not in text
