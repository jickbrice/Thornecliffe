"""The play page is what `scripts/make_play.py` builds, and its scoring agrees with the Python
reward and scorer. The second test needs Node.js (or the `nodejs-bin` package) and is skipped
without it."""

import importlib.util
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from thornecliffe.policy import RETURN_FEE, Decision, grid, owed
from thornecliffe.scoring import reward, score

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "results" / "bench" / "base" / "draw1.jsonl"


def make_play():
    spec = importlib.util.spec_from_file_location("make_play", ROOT / "scripts" / "make_play.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_page_is_the_builder_s():
    base = BASE if BASE.exists() else None
    assert (ROOT / "play.html").read_text(encoding="utf-8") == make_play().build(base)


def node_command() -> list[str] | None:
    if shutil.which("node"):
        return ["node"]
    if importlib.util.find_spec("nodejs") is not None:
        return [sys.executable, "-m", "nodejs"]
    return None


def test_the_page_scores_as_the_python_scorer(tmp_path):
    cmd = node_command()
    if cmd is None:
        pytest.skip("no Node.js")
    page = (ROOT / "play.html").read_text(encoding="utf-8")
    script = re.search(r"const CLAIM_FEE.*?\nconst \$ =", page, re.S).group(0)[: -len("const $ =")]
    cases = []
    for t in grid():
        o = owed(t)
        for clause in (o.clause, 7):
            for paid in sorted({0.0, float(t.price), t.price / 2, float(t.price - RETURN_FEE),
                                o.paid}):
                for party in ("firm", "courier"):
                    d = Decision(clause, paid, party)
                    s = score(t, d)
                    cases.append({
                        "c": {"price": t.price, "owed": {"clause": o.clause, "refund": o.paid,
                                                         "charge_to": o.bearer}},
                        "d": {"clause": clause, "paid": paid, "party": party},
                        "want": [round(reward(t, d), 9), s["right"], s["false_claim"],
                                 round(s["billed_wrongly"] * t.price, 6)]})
    js = tmp_path / "check.js"
    js.write_text(script + "\nconst cases = " + json.dumps(cases) + ";\nlet bad = 0;\n"
                  "for (const k of cases) { const s = score(k.c, k.d);\n"
                  "  const got = [Math.round(reward(k.c, k.d, false) * 1e9) / 1e9, s.right,\n"
                  "               s.falseClaim, Math.round(s.billed * 1e6) / 1e6];\n"
                  "  if (JSON.stringify(got) !== JSON.stringify(k.want)) { bad++;\n"
                  "    if (bad < 4) console.log(JSON.stringify([k, got])); } }\n"
                  "console.log('BAD ' + bad + ' OF ' + cases.length);\n", encoding="utf-8")
    out = subprocess.run([*cmd, str(js)], capture_output=True, text=True, timeout=120).stdout
    assert re.search(r"BAD 0 OF \d+", out), out[-2000:]
