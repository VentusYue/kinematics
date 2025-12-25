#!/usr/bin/env python3
"""
Benchmark old vs fast route collectors (CoinRun/Procgen).

Creates a fresh timestamped experiment directory:
  experiments/bench_collect_<timestamp>/
    old/{data,logs}/...
    fast/{data,logs}/...

Then runs:
  - eval/collect_meta_routes.py
  - eval/collect_meta_routes_fast.py

with identical arguments and prints a small summary.

This is designed so you can run ONE command and paste the output back.
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class RunResult:
    name: str
    cmd: str
    returncode: int
    wall_s: float
    traj_per_s: Optional[float]
    steps_per_s: Optional[float]
    log_path: str


_RE_SPEED = re.compile(r"Speed:\s+([0-9]*\.?[0-9]+)\s+traj/s,\s+([0-9]+)\s+steps/s")


def _parse_speed_from_log(log_text: str) -> Tuple[Optional[float], Optional[float]]:
    m = None
    for m in _RE_SPEED.finditer(log_text):
        pass
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def _run_and_tee(cmd_list, cwd: str, log_path: str, env: Dict[str, str]) -> Tuple[int, float]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    start = time.perf_counter()
    with open(log_path, "w") as f:
        f.write(f"$ {shlex.join(cmd_list)}\n\n")
        f.flush()
        p = subprocess.Popen(
            cmd_list,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )
        assert p.stdout is not None
        for line in p.stdout:
            sys.stdout.write(line)
            f.write(line)
        rc = p.wait()
    wall = time.perf_counter() - start
    return rc, wall


def main():
    p = argparse.ArgumentParser("bench_collect_meta_routes")
    p.add_argument("--env_name", default="coinrun")
    p.add_argument("--model_ckpt", required=True)
    p.add_argument("--device", default="cuda:0")

    # Fast defaults for quick A/B testing
    p.add_argument("--num_tasks", type=int, default=4)
    p.add_argument("--num_processes", type=int, default=4)
    p.add_argument("--adapt_episodes", type=int, default=2)
    p.add_argument("--record_episodes", type=int, default=1)
    p.add_argument("--seed_offset", type=int, default=0)
    p.add_argument("--max_steps", type=int, default=64)
    p.add_argument("--max_ep_len", type=int, default=32)
    p.add_argument("--distribution_mode", default="easy")
    p.add_argument("--require_success", type=int, default=1)
    p.add_argument("--batch_completion_threshold", type=float, default=0.9)
    p.add_argument("--task_timeout", type=int, default=0)
    p.add_argument("--early_abort", type=int, default=1)

    p.add_argument("--ckpt_shard_size", type=int, default=5)
    p.add_argument("--ckpt_flush_secs", type=float, default=10.0)

    # Keep benchmarking stable/reproducible
    p.add_argument("--bench_seed", type=int, default=0, help="Fixed seed for reproducible A/B runs")
    p.add_argument("--deterministic_policy", type=int, default=1, help="Force deterministic actions in both phases")
    p.add_argument("--use_compile", type=int, default=0, help="0 recommended for micro-bench (reduces variability)")
    p.add_argument("--use_amp", type=int, default=1)
    p.add_argument("--adapt_deterministic", type=int, default=None, help="Override adapt_deterministic (default: from deterministic_policy)")
    p.add_argument("--record_deterministic", type=int, default=None, help="Override record_deterministic (default: from deterministic_policy)")

    p.add_argument(
        "--base_out_dir",
        default="/root/backup/kinematics/experiments",
        help="Where to create the benchmark experiment directory",
    )
    p.add_argument("--skip_old", action="store_true", help="Skip running the old collector")
    p.add_argument("--skip_fast", action="store_true", help="Skip running the fast collector")
    p.add_argument("--tag", default="", help="Optional suffix in experiment name")
    args = p.parse_args()

    # ----------------------------
    # Reproducibility knobs
    # ----------------------------
    # These make the policy action sequence stable across runs (and thus env trajectories too,
    # since procgen start_level seeds are fixed). We keep this lightweight for benchmarking.
    os.environ.setdefault("PYTHONHASHSEED", str(args.bench_seed))
    try:
        import numpy as _np
        _np.random.seed(args.bench_seed)
    except Exception:
        pass
    try:
        import torch as _torch
        _torch.manual_seed(args.bench_seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(args.bench_seed)
    except Exception:
        pass

    if args.adapt_deterministic is None:
        args.adapt_deterministic = 1 if int(args.deterministic_policy) else 0
    if args.record_deterministic is None:
        args.record_deterministic = 1 if int(args.deterministic_policy) else 1

    ts = time.strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    exp_root = os.path.join(args.base_out_dir, f"bench_collect_{ts}{tag}")

    def paths(which: str):
        out_dir = os.path.join(exp_root, which)
        data_dir = os.path.join(out_dir, "data")
        logs_dir = os.path.join(out_dir, "logs")
        ckpt_dir = os.path.join(data_dir, "ckpt")
        routes_npz = os.path.join(data_dir, "routes.npz")
        log_path = os.path.join(logs_dir, "collection.log")
        return out_dir, data_dir, logs_dir, ckpt_dir, routes_npz, log_path

    old_out, _, _, old_ckpt, old_routes, old_log = paths("old")
    fast_out, _, _, fast_ckpt, fast_routes, fast_log = paths("fast")
    faster_out, _, _, faster_ckpt, faster_routes, faster_log = paths("faster")

    os.makedirs(exp_root, exist_ok=True)
    os.makedirs(old_out, exist_ok=True)
    os.makedirs(fast_out, exist_ok=True)
    os.makedirs(faster_out, exist_ok=True)

    repo_cwd = "/root/backup/kinematics"

    base_env = os.environ.copy()
    # Keep output cleaner + reduce warning spam in workers
    base_env.setdefault("PYTHONWARNINGS", "ignore::DeprecationWarning:gym")
    base_env.setdefault("GYM_LOGGER_LEVEL", "error")
    base_env.setdefault("GYM_DISABLE_WARNINGS", "1")
    base_env.setdefault("PYTHONHASHSEED", str(args.bench_seed))

    common_args = [
        "--env_name", args.env_name,
        "--model_ckpt", args.model_ckpt,
        "--device", args.device,
        "--num_tasks", str(args.num_tasks),
        "--num_processes", str(args.num_processes),
        "--adapt_episodes", str(args.adapt_episodes),
        "--record_episodes", str(args.record_episodes),
        "--adapt_deterministic", str(int(args.adapt_deterministic)),
        "--record_deterministic", str(int(args.record_deterministic)),
        "--seed_offset", str(args.seed_offset),
        "--max_steps", str(args.max_steps),
        "--max_ep_len", str(args.max_ep_len),
        "--distribution_mode", args.distribution_mode,
        "--require_success", str(args.require_success),
        "--batch_completion_threshold", str(args.batch_completion_threshold),
        "--task_timeout", str(args.task_timeout),
        "--early_abort", str(args.early_abort),
        "--xy_fail_policy", "warn_only",
        "--ckpt_shard_size", str(args.ckpt_shard_size),
        "--ckpt_flush_secs", str(args.ckpt_flush_secs),
        "--use_compile", str(args.use_compile),
        "--use_amp", str(args.use_amp),
        "--resume",
    ]

    runs = []

    old_cmd = ["python", "-W", "ignore", "eval/collect_meta_routes.py",
               "--out_npz", old_routes, "--ckpt_dir", old_ckpt] + common_args
    fast_cmd = ["python", "-W", "ignore", "eval/collect_meta_routes_fast.py",
                "--out_npz", fast_routes, "--ckpt_dir", fast_ckpt] + common_args
    faster_cmd = ["python", "-W", "ignore", "eval/collect_meta_routes_faster.py",
                "--out_npz", faster_routes, "--ckpt_dir", faster_ckpt] + common_args

    print("=" * 80)
    print("BENCH: old vs fast vs faster")
    print("=" * 80)
    print(f"Experiment dir: {exp_root}")
    if not args.skip_old:
        print(f"Old   log: {old_log}")
    if not args.skip_fast:
        print(f"Fast  log: {fast_log}")
    print(f"Faster log: {faster_log}")
    print("-" * 80)
    
    if not args.skip_old:
        print("Old command:")
        print("  " + shlex.join(old_cmd))
    if not args.skip_fast:
        print("Fast command:")
        print("  " + shlex.join(fast_cmd))
    print("Faster command:")
    print("  " + shlex.join(faster_cmd))
    print("=" * 80)
    print("")

    bench_targets = []
    if not args.skip_old:
        bench_targets.append(("old", old_cmd, old_log))
    if not args.skip_fast:
        bench_targets.append(("fast", fast_cmd, fast_log))
    bench_targets.append(("faster", faster_cmd, faster_log))

    for name, cmd, log_path in bench_targets:
        print("")
        print("=" * 80)
        print(f"RUN: {name}")
        print("=" * 80)
        rc, wall = _run_and_tee(cmd, cwd=repo_cwd, log_path=log_path, env=base_env)
        try:
            with open(log_path, "r") as f:
                text = f.read()
        except Exception:
            text = ""
        traj_s, steps_s = _parse_speed_from_log(text)
        runs.append(RunResult(name=name, cmd=shlex.join(cmd), returncode=rc, wall_s=wall,
                              traj_per_s=traj_s, steps_per_s=steps_s, log_path=log_path))
        print(f"\n[bench] {name} returncode={rc}, wall={wall:.1f}s, parsed_speed=({traj_s},{steps_s})")

    print("")
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Experiment dir: {exp_root}")
    print("")
    for r in runs:
        ts = f"{r.traj_per_s:.2f}" if r.traj_per_s is not None else "NA"
        ss = f"{r.steps_per_s:.0f}" if r.steps_per_s is not None else "NA"
        print(f"- {r.name:4s} | rc={r.returncode} | wall={r.wall_s:6.1f}s | speed={ts:>6} traj/s | {ss:>7} steps/s | log={r.log_path}")

    # Speedup ratio (steps/s) if available
    old = next((x for x in runs if x.name == "old"), None)
    fast = next((x for x in runs if x.name == "fast"), None)
    faster = next((x for x in runs if x.name == "faster"), None)

    if old and fast and old.steps_per_s and fast.steps_per_s and old.steps_per_s > 0:
        print("")
        print(f"[bench] fast   vs old (steps/s): {fast.steps_per_s/old.steps_per_s:.2f}x")
    
    if old and faster and old.steps_per_s and faster.steps_per_s and old.steps_per_s > 0:
        print(f"[bench] faster vs old (steps/s): {faster.steps_per_s/old.steps_per_s:.2f}x")

    if fast and faster and fast.steps_per_s and faster.steps_per_s and fast.steps_per_s > 0:
        print(f"[bench] faster vs fast (steps/s): {faster.steps_per_s/fast.steps_per_s:.2f}x")
    print("=" * 80)

    # Non-zero exit if any run failed (makes it easy to catch in logs)
    for r in runs:
        if r.returncode != 0:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


