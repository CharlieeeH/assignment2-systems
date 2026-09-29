"""End-to-end benchmark for the CS336 basics Transformer.

Examples:
    uv run python scripts/benchmark.py --model-size small --mode all
    uv run python scripts/benchmark.py --model-size small --mode forward-backward
    uv run python scripts/benchmark.py --model-size small --mixed-precision
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from timeit import default_timer
from typing import Literal

import torch

from cs336_basics.model import BasicsTransformerLM


Mode = Literal["forward", "forward-backward", "full"]
LOGGER = logging.getLogger("benchmark")


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


@dataclass(frozen=True)
class BenchmarkResult:
    mode: Mode
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    measurements_ms: list[float]


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work, if any, to finish."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_autocast_context(device: torch.device, mixed_precision: bool):
    if not mixed_precision:
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("BF16 autocast is only supported by this script on CPU or CUDA devices")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def run_step(
    mode: Mode,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
) -> None:
    """Run one benchmark step without synchronizing or measuring it."""
    needs_backward = mode != "forward"
    if needs_backward:
        optimizer.zero_grad(set_to_none=True)

    # Keep autograd enabled for forward-only measurements so this measures the
    # forward part of a training step, rather than inference under no_grad().
    with make_autocast_context(device, mixed_precision):
        logits = model(inputs)
        if needs_backward:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1)
            )

    if needs_backward:
        loss.backward()
        if mode == "full":
            optimizer.step()


def benchmark_mode(
    mode: Mode,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
    warmup_steps: int,
    measurement_steps: int,
) -> BenchmarkResult:
    LOGGER.info(
        "Starting mode=%s warmup_steps=%d measurement_steps=%d",
        mode,
        warmup_steps,
        measurement_steps,
    )
    for step_index in range(warmup_steps):
        run_step(mode, model, optimizer, inputs, targets, device, mixed_precision)
        synchronize(device)
        LOGGER.info("Warm-up mode=%s step=%d/%d complete", mode, step_index + 1, warmup_steps)

    measurements_ms: list[float] = []
    for step_index in range(measurement_steps):
        # The pre-step synchronization ensures no work from setup or a previous
        # mode is included. The post-step synchronization is required because
        # CUDA launches are asynchronous with respect to the CPU timer.
        synchronize(device)
        start = default_timer()
        run_step(mode, model, optimizer, inputs, targets, device, mixed_precision)
        synchronize(device)
        elapsed_ms = (default_timer() - start) * 1_000
        measurements_ms.append(elapsed_ms)
        # Logging happens after timing and synchronization, so file I/O is not
        # included in the measured interval.
        LOGGER.info(
            "Measurement mode=%s step=%d/%d elapsed_ms=%.3f",
            mode,
            step_index + 1,
            measurement_steps,
            elapsed_ms,
        )

    result = BenchmarkResult(
        mode=mode,
        mean_ms=statistics.fmean(measurements_ms),
        std_ms=statistics.pstdev(measurements_ms),
        min_ms=min(measurements_ms),
        max_ms=max(measurements_ms),
        measurements_ms=measurements_ms,
    )
    LOGGER.info(
        "Completed mode=%s mean_ms=%.3f std_ms=%.3f min_ms=%.3f max_ms=%.3f",
        result.mode,
        result.mean_ms,
        result.std_ms,
        result.min_ms,
        result.max_ms,
    )
    return result


def configure_logging(log_file: str, log_level: str) -> Path:
    path = Path(log_file).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    handler = logging.FileHandler(path, mode="a", encoding="utf-8")
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    LOGGER.handlers.clear()
    LOGGER.addHandler(handler)
    LOGGER.setLevel(getattr(logging, log_level))
    LOGGER.propagate = False
    LOGGER.info("========== benchmark run started ==========")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "full", "all"),
        default="all",
        help="full includes forward, backward, and the optimizer step",
    )
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--mixed-precision", action="store_true", help="use BF16 autocast")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    parser.add_argument("--log-file", default="benchmark.log", help="append activity logs to this file")
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args()

    for name in ("vocab_size", "context_length", "batch_size", "warmup_steps", "measurement_steps"):
        minimum = 0 if name == "warmup_steps" else 1
        if getattr(args, name) < minimum:
            parser.error(f"--{name.replace('_', '-')} must be at least {minimum}")
    return args


def main() -> None:
    args = parse_args()
    log_path = configure_logging(args.log_file, args.log_level)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    config = MODEL_CONFIGS[args.model_size].copy()
    for name in ("d_model", "d_ff", "num_layers", "num_heads"):
        override = getattr(args, name)
        if override is not None:
            if override < 1:
                raise ValueError(f"--{name.replace('_', '-')} must be positive")
            config[name] = override
    if config["d_model"] % config["num_heads"] != 0:
        raise ValueError("d_model must be divisible by num_heads")

    LOGGER.info(
        "Configuration model_size=%s device=%s mixed_precision=%s batch_size=%d "
        "context_length=%d vocab_size=%d warmup_steps=%d measurement_steps=%d config=%s",
        args.model_size,
        device,
        args.mixed_precision,
        args.batch_size,
        args.context_length,
        args.vocab_size,
        args.warmup_steps,
        args.measurement_steps,
        config,
    )

    # Creating the model in a device context avoids first materializing very
    # large configurations in host memory and then copying them to the GPU.
    with torch.device(device):
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            **config,
        )
    model.train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info("Model initialized parameters=%d", parameter_count)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    inputs = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )

    modes: list[Mode] = (
        ["forward", "forward-backward", "full"] if args.mode == "all" else [args.mode]
    )
    results = [
        benchmark_mode(
            mode=mode,
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            device=device,
            mixed_precision=args.mixed_precision,
            warmup_steps=args.warmup_steps,
            measurement_steps=args.measurement_steps,
        )
        for mode in modes
    ]
    LOGGER.info("All requested modes completed")

    if args.json:
        print(
            json.dumps(
                {
                    "configuration": {
                        "model_size": args.model_size,
                        **config,
                        "vocab_size": args.vocab_size,
                        "context_length": args.context_length,
                        "batch_size": args.batch_size,
                        "device": str(device),
                        "mixed_precision": args.mixed_precision,
                        "warmup_steps": args.warmup_steps,
                        "measurement_steps": args.measurement_steps,
                    },
                    "results": [asdict(result) for result in results],
                },
                indent=2,
            )
        )
        LOGGER.info("Results emitted as JSON log_file=%s", log_path)
        return

    precision = "BF16 autocast" if args.mixed_precision else "FP32"
    print(
        f"model={args.model_size} device={device} precision={precision} "
        f"batch={args.batch_size} context={args.context_length}"
    )
    print(f"{'mode':<18} {'mean (ms)':>12} {'std (ms)':>12} {'min (ms)':>12} {'max (ms)':>12}")
    for result in results:
        print(
            f"{result.mode:<18} {result.mean_ms:>12.3f} {result.std_ms:>12.3f} "
            f"{result.min_ms:>12.3f} {result.max_ms:>12.3f}"
        )
    print(f"activity log: {log_path}")
    LOGGER.info("Results emitted as table log_file=%s", log_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Benchmark failed")
        raise
