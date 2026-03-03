"""
Batch launcher for simulation Python scripts with per-run report.

Expected script behavior:
- Print "OUT_DIR=<path>" when finished.
- Print "RUN_CONFIG_PATH=<path_to_run_config.json>" when finished.

This launcher also tries sensible fallbacks if markers are missing.
"""

import argparse
import json
import re
import shlex
import subprocess
import time
from pathlib import Path


PARAM_KEYS = [
    "measurement",
    "N",
    "state",
    "states",
    "gamma",
    "omega",
    "phi1",
    "phi2",
    "theta",
    "phi_state",
    "etas",
    "eta_minus_list",
    "eta_vector",
    "unitary_mode",
    "ntraj",
    "chunk_size",
    "t_end_T1",
    "dt_T1",
    "runtime_sec",
]


def parse_marker(text: str, key: str):
    pattern = rf"^{re.escape(key)}=(.+)$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return None
    return match.group(1).strip()


def safe_name(name: str):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def compact_value(value):
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        if len(value) <= 8:
            return ",".join(str(x) for x in value)
        return f"{value[:4]}... (len={len(value)})"
    if isinstance(value, dict):
        keys = list(value.keys())
        if len(keys) <= 8:
            return ",".join(str(k) for k in keys)
        return f"{keys[:4]}... (len={len(keys)})"
    return str(value)


def summarize_meta(meta: dict):
    out = {}
    for key in PARAM_KEYS:
        if key in meta:
            out[key] = compact_value(meta[key])
    return out


def parse_done_out_dir(stdout_text: str):
    match = re.search(r"Done\. Results saved in:\s*(.+)$", stdout_text, flags=re.MULTILINE)
    if match:
        return match.group(1).strip()
    return None


def ensure_abs_path(path_text: str, repo_root: Path):
    p = Path(path_text)
    if p.is_absolute():
        return p.resolve()
    return (repo_root / p).resolve()


def to_repo_subpath(path_text: str, repo_root: Path):
    if not path_text:
        return ""
    try:
        p = ensure_abs_path(path_text, repo_root=repo_root)
        return p.resolve().relative_to(repo_root.resolve()).as_posix()
    except Exception:
        return str(path_text)


def sanitize_text_paths(text: str, repo_root: Path):
    if not text:
        return ""
    root = str(repo_root.resolve())
    root_fwd = root.replace("\\", "/")
    out = text
    for prefix in [root + "\\", root + "/", root_fwd + "/", root_fwd + "\\", root, root_fwd]:
        out = out.replace(prefix, "")
    return out


def to_minutes(seconds: float):
    return float(seconds) / 60.0


def to_hours(seconds: float):
    return float(seconds) / 3600.0


def run_one(
    run_cfg: dict,
    python_exec: str,
    repo_root: Path,
    report_dir: Path,
    timeout_sec: int | None,
):
    name = run_cfg.get("name")
    script = run_cfg.get("script")
    args = run_cfg.get("args", [])
    enabled = run_cfg.get("enabled", True)

    if not name:
        name = Path(script).stem if script else "unnamed_run"

    row = {
        "name": name,
        "script": script,
        "status": "skipped",
        "exit_code": None,
        "duration_sec": 0.0,
        "out_dir": "",
        "run_config_path": "",
    }

    if not enabled:
        row["status"] = "disabled"
        return row

    if not script:
        row["status"] = "invalid_config"
        row["error"] = "missing script path"
        return row

    script_path = ensure_abs_path(script, repo_root=repo_root)
    if not script_path.exists():
        row["status"] = "missing_script"
        row["error"] = f"script not found: {to_repo_subpath(str(script_path), repo_root=repo_root)}"
        return row

    cmd_exec = [python_exec, str(script_path), *[str(x) for x in args]]
    cmd_show = [python_exec, script, *[str(x) for x in args]]
    row["cmd"] = " ".join(shlex.quote(c) for c in cmd_show)

    t0 = time.perf_counter()
    timeout_hit = False
    try:
        proc = subprocess.run(
            cmd_exec,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        exit_code = int(proc.returncode)
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        timeout_hit = True
        exit_code = 124
        stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""

    duration = float(time.perf_counter() - t0)
    duration_min = to_minutes(duration)
    duration_hr = to_hours(duration)

    stdout_clean = sanitize_text_paths(stdout_text, repo_root=repo_root)
    stderr_clean = sanitize_text_paths(stderr_text, repo_root=repo_root)

    log_path = report_dir / f"{safe_name(name)}.log"
    log_path.write_text(
        "\n".join(
            [
                f"NAME: {name}",
                f"SCRIPT: {to_repo_subpath(str(script_path), repo_root=repo_root)}",
                f"CMD: {row['cmd']}",
                f"EXIT_CODE: {exit_code}",
                f"DURATION_SEC: {duration:.6f}",
                f"DURATION_MIN: {duration_min:.6f}",
                f"DURATION_HR: {duration_hr:.6f}",
                "",
                "===== STDOUT =====",
                stdout_clean,
                "",
                "===== STDERR =====",
                stderr_clean,
            ]
        ),
        encoding="utf-8",
    )

    out_dir = parse_marker(stdout_text, "OUT_DIR")
    run_cfg_path = parse_marker(stdout_text, "RUN_CONFIG_PATH")

    if not out_dir:
        done_out_dir = parse_done_out_dir(stdout_text)
        if done_out_dir:
            out_dir = str(ensure_abs_path(done_out_dir, repo_root=repo_root))

    if not run_cfg_path and out_dir:
        candidate = Path(out_dir) / "run_config.json"
        if candidate.exists():
            run_cfg_path = str(candidate.resolve())

    meta = {}
    if run_cfg_path:
        cfg_path = ensure_abs_path(run_cfg_path, repo_root=repo_root)
        if cfg_path.exists():
            try:
                meta = json.loads(cfg_path.read_text(encoding="utf-8"))
                run_cfg_path = str(cfg_path.resolve())
            except json.JSONDecodeError:
                meta = {}

    row.update(
        {
            "status": "timeout" if timeout_hit else ("ok" if exit_code == 0 else "failed"),
            "exit_code": exit_code,
            "duration_sec": duration,
            "duration_min": duration_min,
            "duration_hr": duration_hr,
            "out_dir": to_repo_subpath(out_dir or "", repo_root=repo_root),
            "run_config_path": to_repo_subpath(run_cfg_path or "", repo_root=repo_root),
            "log_path": to_repo_subpath(str(log_path), repo_root=repo_root),
        }
    )
    row.update(summarize_meta(meta))

    if exit_code != 0 and stderr_text.strip():
        row["error"] = stderr_text.strip().splitlines()[-1]

    return row


def make_parser():
    parser = argparse.ArgumentParser(description="Run multiple simulation scripts and build a short report")
    parser.add_argument(
        "--config",
        type=str,
        default=r".\Codes\sim_batch_example.json",
        help="Path to JSON batch config",
    )
    parser.add_argument(
        "--python",
        type=str,
        default="",
        help="Python executable used to launch scripts (overrides config)",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Optional comma-separated run names to execute",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="",
        help="Optional override for report output directory",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=0,
        help="Per-run timeout in seconds (0 disables timeout)",
    )
    return parser


def main():
    args = make_parser().parse_args()
    repo_root = Path.cwd()
    config_path = ensure_abs_path(args.config, repo_root=repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    runs = config.get("runs", [])
    if not runs:
        raise ValueError("Config contains no runs.")

    python_exec = args.python.strip() if args.python else str(config.get("python", "python"))
    timeout_cfg = int(config.get("timeout_sec", 0))
    timeout_sec = int(args.timeout_sec) if int(args.timeout_sec) > 0 else timeout_cfg
    timeout_sec = None if timeout_sec <= 0 else timeout_sec

    report_dir_cfg = config.get("report_dir", r".\Codes\BatchReports")
    report_dir = (
        ensure_abs_path(args.report_dir, repo_root=repo_root)
        if args.report_dir
        else ensure_abs_path(report_dir_cfg, repo_root=repo_root)
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    only = {x.strip() for x in args.only.split(",") if x.strip()}

    rows = []
    for run_cfg in runs:
        name = run_cfg.get("name", "")
        if only and name not in only:
            continue
        print(f"Running: {name or run_cfg.get('script', '<unnamed>')}")
        row = run_one(
            run_cfg=run_cfg,
            python_exec=python_exec,
            repo_root=repo_root,
            report_dir=report_dir,
            timeout_sec=timeout_sec,
        )
        rows.append(row)
        print(
            f"  -> status={row['status']} exit={row['exit_code']} "
            f"time={row['duration_sec']:.2f}s ({row['duration_min']:.2f}m, {row['duration_hr']:.3f}h)"
        )

    if not rows:
        print("No runs selected.")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = report_dir / f"batch_report_{ts}.json"

    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    ok_count = sum(1 for r in rows if r.get("status") == "ok")
    fail_count = sum(1 for r in rows if r.get("status") in {"failed", "timeout"})
    print("")
    print(f"Completed runs: {len(rows)}")
    print(f"OK: {ok_count}")
    print(f"Failed/Timeout: {fail_count}")
    print(f"JSON report: {to_repo_subpath(str(json_path), repo_root=repo_root)}")

    if fail_count > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
