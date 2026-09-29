"""End-to-end benchmark for the CS336 basics Transformer (handout §2.1).

Examples:
    uv run python scripts/benchmark.py --model-size small --mode all
    uv run python scripts/benchmark.py --model-size small --mode forward-backward
    uv run python scripts/benchmark.py --model-size small --mixed-precision
    uv run python scripts/benchmark.py --model-size small --warmup-steps 0

Nsight Systems (§2.1.4). Warm-up and measurement are wrapped in NVTX ranges
named "warmup" and "measure", and each step is split into "forward",
"backward" and "optimizer_step". Capture only the measured steps with:
    uv run nsys profile -o small_ctx512 --trace=cuda,nvtx,osrt \\
        --pytorch=functions-trace,autograd-shapes-nvtx \\
        --capture-range=nvtx --nvtx-capture=measure --capture-range-end=stop \\
        --env-var=NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \\
        -- python scripts/benchmark.py --model-size small --mode full --annotate-attention

Memory profiling (§2.1.6). Records allocations during the measured steps and
writes one snapshot per mode for https://pytorch.org/memory_viz:
    uv run python scripts/benchmark.py --model-size small --mode forward \\
        --measurement-steps 1 --memory-profile --memory-snapshot-dir snapshots

CUDA out-of-memory errors are caught and reported per mode (status "oom")
instead of crashing, so sweeps over large configurations keep going.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import statistics
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from timeit import default_timer
from typing import Literal

import torch
from einops import einsum

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import softmax
from cs336_basics.optimizer import AdamW as BasicsAdamW


Mode = Literal["forward", "forward-backward", "full"]
Phase = Literal["forward", "backward", "optimizer_step"]
LOGGER = logging.getLogger("benchmark")


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}

MODE_PHASES: dict[Mode, tuple[Phase, ...]] = {
    "forward": ("forward",),
    "forward-backward": ("forward", "backward"),
    "full": ("forward", "backward", "optimizer_step"),
}


@dataclass(frozen=True)
class PhaseStats:
    mean_ms: float
    std_ms: float


@dataclass(frozen=True)
class BenchmarkResult:
    mode: Mode
    status: Literal["ok", "oom"] = "ok"
    mean_ms: float | None = None
    std_ms: float | None = None
    min_ms: float | None = None
    max_ms: float | None = None
    measurements_ms: list[float] = field(default_factory=list)
    # Per-phase breakdown of the same steps. On CUDA this uses events, so the
    # total above is measured without any extra host synchronization.
    phases: dict[str, PhaseStats] = field(default_factory=dict)
    peak_memory_mib: float | None = None
    memory_snapshot: str | None = None


def synchronize(device: torch.device) -> None:
    """Wait for queued CUDA work, if any, to finish."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def nvtx_range(name: str, device: torch.device):
    """NVTX ranges for nsys; a no-op off CUDA so the script also runs on CPU."""
    if device.type == "cuda":
        return torch.cuda.nvtx.range(name)
    return nullcontext()


def make_autocast_context(device: torch.device, mixed_precision: bool):
    if not mixed_precision:
        return nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("BF16 autocast is only supported by this script on CPU or CUDA devices")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """cs336_basics.model.scaled_dot_product_attention with NVTX ranges (§2.1.4)."""
    with torch.cuda.nvtx.range("scaled dot product attention"):
        with torch.cuda.nvtx.range("computing attention scores"):
            d_k = K.shape[-1]
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with torch.cuda.nvtx.range("computing softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with torch.cuda.nvtx.range("final matmul"):
            return einsum(attention_weights, V, "... query key, ... key d_v ->  ... query d_v")


class PhaseTimer:
    """Marks phase boundaries inside one step; read the durations after a sync."""

    def __init__(self, device: torch.device) -> None:
        self.use_events = device.type == "cuda"
        self.marks: list[tuple[str, torch.cuda.Event | float]] = []

    def mark(self, name: str) -> None:
        if self.use_events:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks.append((name, event))
        else:
            self.marks.append((name, default_timer()))

    def durations_ms(self) -> dict[str, float]:
        durations = {}
        for (_, start), (name, end) in zip(self.marks, self.marks[1:]):
            if self.use_events:
                durations[name] = start.elapsed_time(end)
            else:
                durations[name] = (end - start) * 1_000
        return durations


def run_step(
    mode: Mode,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    mixed_precision: bool,
    timer: PhaseTimer | None = None,
) -> None:
    """Run one benchmark step without synchronizing or measuring it."""
    mark: Callable[[str], None] = timer.mark if timer is not None else (lambda _: None)
    needs_backward = mode != "forward"
    if needs_backward:
        optimizer.zero_grad(set_to_none=True)

    mark("start")
    # Keep autograd enabled for forward-only measurements so this measures the
    # forward part of a training step, rather than inference under no_grad().
    with nvtx_range("forward", device), make_autocast_context(device, mixed_precision):
        logits = model(inputs)
        if needs_backward:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    mark("forward")

    if needs_backward:
        with nvtx_range("backward", device):
            loss.backward()
        mark("backward")
        if mode == "full":
            with nvtx_range("optimizer_step", device):
                optimizer.step()
            mark("optimizer_step")


def free_memory(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    optimizer.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


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
    memory_snapshot_path: Path | None = None,
) -> BenchmarkResult:
    LOGGER.info(
        "Starting mode=%s warmup_steps=%d measurement_steps=%d",
        mode,
        warmup_steps,
        measurement_steps,
    )
    recording_memory = False
    try:
        with nvtx_range(mode, device):
            with nvtx_range("warmup", device):
                for step_index in range(warmup_steps):
                    run_step(mode, model, optimizer, inputs, targets, device, mixed_precision)
                    synchronize(device)
                    LOGGER.info("Warm-up mode=%s step=%d/%d complete", mode, step_index + 1, warmup_steps)

            if device.type == "cuda":
                synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            if memory_snapshot_path is not None:
                torch.cuda.memory._record_memory_history(max_entries=1_000_000)
                recording_memory = True

            measurements_ms: list[float] = []
            phase_ms: dict[str, list[float]] = {phase: [] for phase in MODE_PHASES[mode]}
            with nvtx_range("measure", device):
                for step_index in range(measurement_steps):
                    # The pre-step synchronization ensures no work from setup or a previous
                    # mode is included. The post-step synchronization is required because
                    # CUDA launches are asynchronous with respect to the CPU timer.
                    timer = PhaseTimer(device)
                    synchronize(device)
                    with nvtx_range(f"step_{step_index}", device):
                        start = default_timer()
                        run_step(mode, model, optimizer, inputs, targets, device, mixed_precision, timer)
                        synchronize(device)
                        elapsed_ms = (default_timer() - start) * 1_000
                    measurements_ms.append(elapsed_ms)
                    step_phases = timer.durations_ms()
                    for phase, duration in step_phases.items():
                        phase_ms[phase].append(duration)
                    # Logging happens after timing and synchronization, so file I/O is not
                    # included in the measured interval.
                    LOGGER.info(
                        "Measurement mode=%s step=%d/%d elapsed_ms=%.3f phases_ms=%s",
                        mode,
                        step_index + 1,
                        measurement_steps,
                        elapsed_ms,
                        {phase: round(duration, 3) for phase, duration in step_phases.items()},
                    )
    except torch.OutOfMemoryError:
        LOGGER.warning("Out of memory in mode=%s", mode, exc_info=True)
        snapshot = dump_memory_snapshot(memory_snapshot_path) if recording_memory else None
        free_memory(optimizer, device)
        return BenchmarkResult(mode=mode, status="oom", memory_snapshot=snapshot)

    peak_memory_mib = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
    snapshot = dump_memory_snapshot(memory_snapshot_path) if recording_memory else None

    result = BenchmarkResult(
        mode=mode,
        mean_ms=statistics.fmean(measurements_ms),
        std_ms=statistics.pstdev(measurements_ms),
        min_ms=min(measurements_ms),
        max_ms=max(measurements_ms),
        measurements_ms=measurements_ms,
        phases={phase: PhaseStats(statistics.fmean(values), statistics.pstdev(values)) for phase, values in phase_ms.items()},
        peak_memory_mib=peak_memory_mib,
        memory_snapshot=snapshot,
    )
    LOGGER.info(
        "Completed mode=%s mean_ms=%.3f std_ms=%.3f min_ms=%.3f max_ms=%.3f peak_memory_mib=%s",
        result.mode,
        result.mean_ms,
        result.std_ms,
        result.min_ms,
        result.max_ms,
        None if peak_memory_mib is None else round(peak_memory_mib, 1),
    )
    free_memory(optimizer, device)
    return result


def dump_memory_snapshot(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._dump_snapshot(str(path))
    torch.cuda.memory._record_memory_history(enabled=None)
    LOGGER.info("Memory snapshot written path=%s", path)
    return str(path)


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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
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
    parser.add_argument(
        "--optimizer",
        choices=("basics", "torch"),
        default="basics",
        help="basics = cs336_basics.optimizer.AdamW (the handout's 'your AdamW'), torch = torch.optim.AdamW",
    )
    parser.add_argument("--mixed-precision", action="store_true", help="use BF16 autocast")
    parser.add_argument(
        "--annotate-attention",
        action="store_true",
        help="swap in an NVTX-annotated scaled_dot_product_attention for nsys",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="record CUDA allocations during the measured steps and dump a snapshot per mode",
    )
    parser.add_argument("--memory-snapshot-dir", default="memory_snapshots")
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
    if args.memory_profile and not args.device.startswith("cuda"):
        parser.error("--memory-profile requires a CUDA device")
    if args.annotate_attention and not args.device.startswith("cuda"):
        parser.error("--annotate-attention requires a CUDA device")
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

    if args.annotate_attention:
        # model.py looks the function up as a module global at call time.
        cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

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
        "Configuration model_size=%s device=%s mixed_precision=%s batch_size=%d context_length=%d vocab_size=%d warmup_steps=%d measurement_steps=%d optimizer=%s config=%s",
        args.model_size,
        device,
        args.mixed_precision,
        args.batch_size,
        args.context_length,
        args.vocab_size,
        args.warmup_steps,
        args.measurement_steps,
        args.optimizer,
        config,
    )

    modes: list[Mode] = ["forward", "forward-backward", "full"] if args.mode == "all" else [args.mode]
    precision_tag = "bf16" if args.mixed_precision else "fp32"
    parameter_count = None
    try:
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
        optimizer_cls = BasicsAdamW if args.optimizer == "basics" else torch.optim.AdamW
        optimizer = optimizer_cls(model.parameters(), lr=args.learning_rate)
        inputs = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
    except torch.OutOfMemoryError:
        LOGGER.warning("Out of memory while initializing the model", exc_info=True)
        results = [BenchmarkResult(mode=mode, status="oom") for mode in modes]
    else:
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
                memory_snapshot_path=(
                    Path(args.memory_snapshot_dir) / f"{args.model_size}_ctx{args.context_length}_{precision_tag}_{mode}.pickle" if args.memory_profile else None
                ),
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
                        "parameters": parameter_count,
                        "vocab_size": args.vocab_size,
                        "context_length": args.context_length,
                        "batch_size": args.batch_size,
                        "device": str(device),
                        "mixed_precision": args.mixed_precision,
                        "optimizer": args.optimizer,
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
        f"model={args.model_size} params={parameter_count} device={device} precision={precision} batch={args.batch_size} context={args.context_length} warmup={args.warmup_steps}"
    )
    header = f"{'mode':<18} {'mean (ms)':>10} {'std (ms)':>10} {'fwd (ms)':>10} {'bwd (ms)':>10} {'opt (ms)':>10} {'peak MiB':>10}"
    print(header)
    for result in results:
        if result.status == "oom":
            print(f"{result.mode:<18} {'OOM':>10}")
            continue
        phase_cells = [f"{result.phases[phase].mean_ms:>10.3f}" if phase in result.phases else f"{'-':>10}" for phase in ("forward", "backward", "optimizer_step")]
        peak = f"{result.peak_memory_mib:>10.1f}" if result.peak_memory_mib is not None else f"{'-':>10}"
        print(f"{result.mode:<18} {result.mean_ms:>10.3f} {result.std_ms:>10.3f} {' '.join(phase_cells)} {peak}")
    for result in results:
        if result.memory_snapshot:
            print(f"memory snapshot ({result.mode}): {result.memory_snapshot}")
    print(f"activity log: {log_path}")
    LOGGER.info("Results emitted as table log_file=%s", log_path)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOGGER.exception("Benchmark failed")
        raise
