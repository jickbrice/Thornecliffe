"""Write the dataset files in ``data/`` from the generator.

The files are committed; this script rebuilds them byte for byte (`tests/test_data.py` checks
that it does).

    uv run python scripts/make_data.py
"""

from __future__ import annotations

from thornecliffe import test_only
from thornecliffe.data import (
    DATA,
    DRAW_SEED,
    DRAWS,
    SEEDS,
    bench_path,
    row,
    test_only_path,
    train_path,
    write,
)
from thornecliffe.tickets import bench, train_cases


def build() -> dict[str, list[dict]]:
    files = {}
    for seed in SEEDS:
        files[train_path(seed).name] = [
            row(c, split="train", seed=seed, step=s, slot=i, id=f"train-s{seed}-{s:02d}-{i:02d}")
            for s, cs in enumerate(train_cases(seed), 1) for i, c in enumerate(cs)]
    for draw in DRAWS:
        g = DRAW_SEED[draw]
        files[bench_path(draw).name] = [
            row(c, split="bench", draw=draw, cell=cell, id=f"bench-d{draw}-{k:03d}")
            for k, (cell, c) in enumerate(bench(g))]
        files[test_only_path(draw).name] = [
            row(c, split="test_only", draw=draw, cell="wrong item", id=f"wrong-d{draw}-{k:02d}")
            for k, c in enumerate(test_only.cases(g))]
    return files


def main() -> None:
    DATA.mkdir(exist_ok=True)
    for name, rows in build().items():
        write(DATA / name, rows)
        print(f"{name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
