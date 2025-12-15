#!/usr/bin/env python3
"""
Minimal GPU/CPU keepalive loop for Volcano Engine dev machines.

The goal is to generate a tiny, periodic amount of load so that the
platform does not classify the machine as idle and auto-shutdown it.

Usage (recommended):

    # Run in the background with logs:
    cd /root/workspace/ede
    nohup python -u keepalive.py --interval-seconds 300 > keepalive.log 2>&1 &

    # Or pin to a specific GPU:
    CUDA_VISIBLE_DEVICES=0 nohup python -u keepalive.py --interval-seconds 300 > keepalive.log 2>&1 &

You can stop it at any time with:

    pkill -f keepalive.py

Notes:
  - By default this script prefers GPU if PyTorch + CUDA are available.
  - If PyTorch is not available, it falls back to a tiny CPU workload.
  - The workload is intentionally very small relative to typical training jobs.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from typing import Optional


def _setup_logging(verbosity: int) -> None:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        stream=sys.stdout,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal GPU/CPU keepalive loop to prevent idle shutdown.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--interval-seconds",
        type=float,
        default=300.0,
        help="Sleep interval between tiny workloads. Increase to reduce overhead.",
    )
    parser.add_argument(
        "--matrix-size",
        type=int,
        default=1024,
        help="Size of the square matrix used for the tiny workload.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Force CPU workload even if GPU is available.",
    )
    parser.add_argument(
        "--verbosity",
        "-v",
        action="count",
        default=0,
        help="Increase logging verbosity (can be used multiple times).",
    )
    return parser.parse_args()


def _get_torch_device(cpu_only: bool) -> tuple[Optional[str], bool]:
    """
    Decide which device to use and whether torch is available.

    Returns:
        (device, has_torch)
        device: "cuda" | "cpu" | None
        has_torch: whether torch was successfully imported
    """
    try:
        import torch  # type: ignore

        if cpu_only:
            return "cpu", True

        if torch.cuda.is_available():
            return "cuda", True

        logging.warning("CUDA is not available; falling back to CPU workload.")
        return "cpu", True
    except Exception as exc:  # noqa: BLE001
        logging.warning(
            "PyTorch not available (%s); using a pure CPU fallback workload.", exc
        )
        return None, False


def _gpu_tick(matrix_size: int, device: str) -> None:
    """Run a tiny GPU or CPU matmul via PyTorch."""
    import torch  # type: ignore

    dev = torch.device(device)
    x = torch.randn((matrix_size, matrix_size), device=dev)
    y = x @ x
    _ = y.sum().item()


def _cpu_fallback_tick(matrix_size: int) -> None:
    """
    Tiny pure-CPU workload that does not depend on any external libraries.

    This is only used if PyTorch is not importable at all.
    """
    # Simple arithmetic loop; intentionally small.
    total = 0.0
    for i in range(matrix_size * 10):
        total += (i % 97) * 0.123
    # Make sure the loop isn't optimized away.
    if total == -1.0:  # pragma: no cover
        print("unreachable", total)


def main() -> None:
    args = _parse_args()
    _setup_logging(args.verbosity)

    if args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be > 0")

    device, has_torch = _get_torch_device(cpu_only=args.cpu_only)

    if has_torch:
        if device == "cuda":
            logging.info("Using tiny GPU workload on CUDA.")
        else:
            logging.info("Using tiny CPU workload via PyTorch.")
    else:
        logging.info("Using pure CPU fallback workload (no PyTorch available).")

    logging.info(
        "Keepalive loop started with interval=%ss, matrix_size=%s, cpu_only=%s",
        args.interval_seconds,
        args.matrix_size,
        args.cpu_only,
    )

    try:
        while True:
            t0 = time.time()
            try:
                if has_torch:
                    _gpu_tick(args.matrix_size, device or "cpu")
                else:
                    _cpu_fallback_tick(args.matrix_size)
            except Exception as exc:  # noqa: BLE001
                logging.exception("Keepalive tick failed: %s", exc)

            elapsed = time.time() - t0
            logging.debug("Keepalive tick completed in %.3fs", elapsed)

            # Don't allow negative or zero sleep if the interval is very small.
            sleep_for = max(0.1, args.interval_seconds - elapsed)
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        logging.info("Keepalive loop interrupted by user; exiting.")


if __name__ == "__main__":
    main()













