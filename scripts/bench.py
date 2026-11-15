"""Evaluate a model on the bench: greedy decisions on both bench draws and the test-only
wrong-item tickets, with no gate.

The untrained model is benched with no ``--adapter``; a trained agent with its adapter. Every
decision is scored by the exact scorer. This is evaluation only: it is the one place the
test-only tickets are read.

    uv run python scripts/bench.py --out results/bench/base
    uv run python scripts/bench.py --adapter runs/<name>/adapter --out results/bench/gated_s0

Writes ``<out>/draw1.jsonl`` and ``<out>/draw2.jsonl`` (280 rows each: 240 bench tickets and 40
wrong-item tickets) and ``<out>/meta.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from thornecliffe import data
from thornecliffe.prompt import MAX_NEW_TOKENS, chat_text
from thornecliffe.scoring import parse, reward, score

MODEL = "google/gemma-3-12b-it"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--adapter", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--n", type=int, default=None, help="only the first n tickets a file (smoke)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    dtype = torch.bfloat16 if args.device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, device_map=args.device)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    pad = tok.pad_token_id
    args.out.mkdir(parents=True, exist_ok=True)
    meta = {"model": args.model, "adapter": str(args.adapter) if args.adapter else None,
            "load_seconds": round(time.time() - t0, 1), "draws": {}}

    for draw in data.DRAWS:
        t1 = time.time()
        rows = data.read(data.bench_path(draw)) + data.read(data.test_only_path(draw))
        if args.n is not None:
            rows = [r for r in rows if r["split"] == "bench"][: args.n] + \
                   [r for r in rows if r["split"] == "test_only"][: args.n]
        texts = [chat_text(tok, r["prompt"]) for r in rows]
        replies = []
        with torch.no_grad():
            for b in range(0, len(texts), args.batch):
                enc = tok(texts[b : b + args.batch], return_tensors="pt", padding=True,
                          add_special_tokens=False).to(model.device)  # fmt: skip
                gen = model.generate(**enc, do_sample=False, temperature=None, top_p=None,
                                     top_k=None, max_new_tokens=MAX_NEW_TOKENS,
                                     pad_token_id=pad)  # fmt: skip
                n = enc["input_ids"].shape[1]
                replies.extend(tok.decode(g[n:], skip_special_tokens=True) for g in gen)
        out_rows = []
        for r, reply in zip(rows, replies, strict=True):
            t, d = data.ticket(r), parse(reply)
            o = data.required(r)
            out_rows.append({
                "id": r["id"], "split": r["split"], "cell": r["cell"], "kind": r["kind"],
                "level": r["level"], "pushy": r["pushy"], "reply": reply,
                "decision": [d.clause, d.paid, d.bearer], "owed": [o.clause, o.paid, o.bearer],
                "reward": reward(t, d), **score(t, d)})
        (args.out / f"draw{draw}.jsonl").write_bytes(
            "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in out_rows).encode("utf-8"))
        meta["draws"][draw] = {"rows": len(out_rows), "seconds": round(time.time() - t1, 1)}
        print(f"draw {draw}: {len(out_rows)} decisions in {time.time() - t1:.0f}s", flush=True)
    (args.out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
