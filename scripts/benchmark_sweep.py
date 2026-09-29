"""Run scripts/benchmark.py over a grid of configurations and build writeup tables.

Each configuration runs in its own subprocess, so a CUDA OOM (or any crash) in
one configuration cannot leak memory into the next one.

Examples:
    # §2.1.3 (b): all sizes, 5 warm-up / 10 measured steps
    uv run python scripts/benchmark_sweep.py --model-sizes small medium large xl 10B

    # §2.1.3 (c): effect of warm-up
    uv run python scripts/benchmark_sweep.py --model-sizes small medium --warmup-steps 0 1 2 5

    # §2.1.5 (c): FP32 vs BF16 autocast
    uv run python scripts/benchmark_sweep.py --model-sizes small medium --precisions fp32 bf16
"""

from __future__ import annotations

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

BENCHMARK = Path(__file__).with_name("benchmark.py")


def run_one(size: str, context_length: int, precision: str, warmup_steps: int, args: argparse.Namespace, extra: list[str]) -> dict:
    command = [
        sys.executable,
        str(BENCHMARK),
        "--json",
        "--model-size",
        size,
        "--context-length",
        str(context_length),
        "--warmup-steps",
        str(warmup_steps),
        "--measurement-steps",
        str(args.measurement_steps),
        "--mode",
        args.mode,
        "--log-file",
        args.log_file,
        *(["--mixed-precision"] if precision == "bf16" else []),
        *extra,
    ]
    print(f"running size={size} ctx={context_length} precision={precision} warmup={warmup_steps}", file=sys.stderr, flush=True)
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        # Anything that is not an in-process OOM (e.g. the driver killing the
        # process) is recorded as an error row rather than aborting the sweep.
        print(completed.stderr[-2000:], file=sys.stderr)
        return {"configuration": None, "results": None, "error": completed.stderr.strip().splitlines()[-1:] or ["failed"]}
    return json.loads(completed.stdout)


def to_rows(size: str, context_length: int, precision: str, warmup_steps: int, payload: dict, modes: list[str]) -> list[dict]:
    base = {"size": size, "context_length": context_length, "precision": precision, "warmup": warmup_steps}
    if payload.get("results") is None:
        return [{**base, "mode": mode, "status": "error"} for mode in modes]
    rows = []
    for result in payload["results"]:
        row = {**base, "mode": result["mode"], "status": result["status"], "mean_ms": result["mean_ms"], "std_ms": result["std_ms"]}
        for phase, stats in result["phases"].items():
            row[f"{phase}_ms"] = stats["mean_ms"]
        row["peak_memory_mib"] = result["peak_memory_mib"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-sizes", nargs="+", default=["small", "medium", "large", "xl", "10B"])
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[512])
    parser.add_argument("--precisions", nargs="+", choices=("fp32", "bf16"), default=["fp32"])
    parser.add_argument("--warmup-steps", nargs="+", type=int, default=[5])
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--mode", choices=("forward", "forward-backward", "full", "all"), default="all")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--name", default="benchmark", help="basename for the csv/md/tex outputs")
    parser.add_argument("--log-file", default="benchmark.log")
    args, extra = parser.parse_known_args()
    modes = ["forward", "forward-backward", "full"] if args.mode == "all" else [args.mode]

    rows = []
    for size, context_length, precision, warmup_steps in itertools.product(args.model_sizes, args.context_lengths, args.precisions, args.warmup_steps):
        payload = run_one(size, context_length, precision, warmup_steps, args, extra)
        rows.extend(to_rows(size, context_length, precision, warmup_steps, payload, modes))

    frame = pd.DataFrame(rows)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / f"{args.name}.csv", index=False)
    numeric = frame.select_dtypes("number").columns.difference(["context_length", "warmup"])
    formatted = frame.copy()
    formatted[numeric] = formatted[numeric].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    formatted = formatted.fillna("")
    (output_dir / f"{args.name}.md").write_text(to_markdown(formatted))
    formatted.to_latex(output_dir / f"{args.name}.tex", index=False)
    print(to_markdown(formatted))
    print(f"\nwrote {output_dir / args.name}.{{csv,md,tex}}", file=sys.stderr)


def to_markdown(frame: pd.DataFrame) -> str:
    # pandas.DataFrame.to_markdown needs the optional tabulate package.
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body]) + "\n"


if __name__ == "__main__":
    main()
