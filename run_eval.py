#!/usr/bin/env python3
"""Local/cloud Docker Harbor runner and resume helper for FrontierHarness Eval v1.1.0."""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent
VENDOR_BIN = ROOT / "vendor" / "bin"
JOBS_DIR = ROOT / "jobs"
HARBOR_HOME = ROOT / ".harbor-home"
PROXY_SCRIPT = ROOT / "eval_proxy.py"
DATASET_REF = "frontier-harness-eval/frontier-harness-eval"
DATASET_VERSION = "v1.1.0"
IMAGE_TAG = "frontier-eval:v1.1.0"
DEFAULT_MODEL = "isok/gpt-6.1-terra"
DEFAULT_EFFORT = "medium"
DEFAULT_MAX_TOKENS = 131072
DEFAULT_PRICE_IN = 1.00
DEFAULT_PRICE_CACHED = 0.05
DEFAULT_PRICE_OUT = 5.00
DEFAULT_PRICE_CACHE_WRITE = 0.0
EFFORT_CHOICES = ("off", "minimal", "low", "medium", "high", "xhigh", "max")


def is_port_listening(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def pick_free_port(preferred: int = 3426) -> int:
    """Pick the preferred port if free, else ask the OS for an ephemeral port."""
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
                return int(sock.getsockname()[1])
            except OSError:
                if port == 0:
                    raise
    raise RuntimeError("Unable to allocate proxy port")


def resolve_proxy_host(mode: str = "auto") -> str:
    """Resolve hostname/IP reachable from Docker containers to host eval_proxy.py."""
    if mode and mode != "auto":
        return mode
    if sys.platform == "darwin":
        return "host.docker.internal"
    try:
        out = subprocess.check_output(
            [
                "docker",
                "network",
                "inspect",
                "bridge",
                "--format",
                "{{range .IPAM.Config}}{{.Gateway}}{{end}}",
            ],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        first = out.split()[0] if out else ""
        if re.fullmatch(r"\d+\.\d+\.\d+\.\d+", first):
            return first
    except (OSError, subprocess.SubprocessError):
        pass
    return "host.docker.internal"


def start_study_proxy(
    upstream_url: str,
    effort: str,
    max_tokens: int,
    force_model: str,
    port: int,
    job_name: str,
    detach: bool,
    usage_log_path: Path,
) -> tuple[subprocess.Popen[str], int]:
    """Start eval_proxy.py so all harnesses hit identical model, effort, and output caps."""
    if not PROXY_SCRIPT.is_file():
        raise FileNotFoundError(f"Missing study proxy script: {PROXY_SCRIPT}")
    log_path = JOBS_DIR / f"{job_name}.proxy.log"
    pid_path = JOBS_DIR / f"{job_name}.proxy.pid"
    cmd = [
        sys.executable,
        str(PROXY_SCRIPT),
        "--listen-host",
        "0.0.0.0",
        "--listen-port",
        str(port),
        "--upstream",
        upstream_url,
        "--effort",
        effort,
        "--max-tokens",
        str(max_tokens),
        "--force-model",
        force_model,
        "--usage-log",
        str(usage_log_path),
        "--pid-file",
        str(pid_path),
    ]
    log_fh = open(log_path, "a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=detach,
    )
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            log_fh.close()
            raise RuntimeError(f"Study proxy exited early; check {log_path}")
        if is_port_listening("127.0.0.1", port):
            log_fh.close()
            return proc, port
        time.sleep(0.1)
    proc.terminate()
    log_fh.close()
    raise RuntimeError(f"Timed out waiting for study proxy on port {port}")


def _target_arches(task_platform: str) -> tuple[str, ...]:
    if task_platform == "auto":
        return ("amd64", "arm64")
    if "arm64" in task_platform or "aarch64" in task_platform:
        return ("arm64",)
    return ("amd64",)


def collect_mizar_uploads(
    task_platform: str,
    mizar_bin: str | None = None,
    upload_rg: bool = True,
) -> dict[str, str]:
    """Select Linux Mizar (and optional ripgrep) binaries matching container arch."""
    arches = _target_arches(task_platform)
    uploads: dict[str, str] = {}
    custom_path = Path(mizar_bin).expanduser().resolve() if mizar_bin else None

    if custom_path and custom_path.is_file():
        if len(arches) != 1:
            raise ValueError(
                "--mizar-bin pointing to a single binary requires --task-platform linux/amd64 or linux/arm64 (not auto)"
            )
        uploads[str(custom_path)] = f"/tmp/mizar-linux-{arches[0]}"
    else:
        base_dir = custom_path if custom_path else VENDOR_BIN
        for arch in arches:
            arch_dir = base_dir / f"linux-{arch}"
            src = (arch_dir / "mizar") if arch_dir.is_dir() else (base_dir / "mizar")
            if not src.is_file():
                raise FileNotFoundError(
                    f"Missing Mizar Linux binary: {src}. Prepare vendor/bin/linux-{arch}/mizar or pass --mizar-bin."
                )
            uploads[str(src)] = f"/tmp/mizar-linux-{arch}"

    if upload_rg:
        for arch in arches:
            src_rg = VENDOR_BIN / f"linux-{arch}" / "rg"
            if not src_rg.is_file():
                raise FileNotFoundError(
                    f"Missing ripgrep Linux binary: {src_rg}. Prepare vendor/bin/linux-{arch}/rg or pass --no-upload-rg."
                )
            uploads[str(src_rg)] = f"/tmp/rg-linux-{arch}"
    return uploads


def agent_route_url(base_url: str, harness_label: str, use_proxy: bool) -> str:
    """Attach /a/<harness_label> before /v1 when routing through eval_proxy.py."""
    if not use_proxy:
        return base_url
    clean = base_url.rstrip("/")
    if clean.endswith("/v1"):
        return f"{clean[:-3]}/a/{harness_label}/v1"
    return f"{clean}/a/{harness_label}/v1"


def build_mizar_agent(
    model: str,
    protocol: str,
    effort: str,
    max_tokens: int,
    base_url: str,
    api_key: str,
    task_platform: str,
    label: str = "mizar",
    version: str = "0.1.5",
    mizar_bin: str | None = None,
    extra_args: str = "",
    upload_rg: bool = True,
    use_proxy: bool = True,
) -> dict[str, Any]:
    """Build Harbor CommandAgent config for Mizar with per-trial proxy tagging."""
    uploads = collect_mizar_uploads(task_platform, mizar_bin=mizar_bin, upload_rg=upload_rg)
    tools_to_install = ("mizar", "rg") if upload_rg else ("mizar",)
    install_lines = [
        'arch="$(uname -m)"',
        'case "$arch" in',
        "  x86_64|amd64) suf=amd64 ;;",
        "  aarch64|arm64) suf=arm64 ;;",
        '  *) echo "Unsupported container arch: $arch" >&2; exit 1 ;;',
        "esac",
        f"for tool in {' '.join(tools_to_install)}; do",
        '  src="/tmp/${tool}-linux-${suf}"',
        '  [ -f "$src" ] || { echo "Missing uploaded binary $src" >&2; exit 1; }',
        '  install -m 755 "$src" "/usr/local/bin/$tool"',
        "done",
        "rm -f /tmp/mizar-linux-* /tmp/rg-linux-*",
        "mizar --version",
    ]
    if upload_rg:
        install_lines.append("rg --version | head -n 1")

    run_lines = [
        'set -a; . "$STATE/agent.env"; set +a',
    ]
    if use_proxy:
        run_lines.extend(
            [
                'mkdir -p "$LOGS"',
                (
                    "TRIAL_ID=\"$(cat /proc/sys/kernel/random/uuid 2>/dev/null "
                    "|| uuidgen 2>/dev/null "
                    "|| printf '%s-%s' \"$(hostname)\" \"$$\")\""
                ),
                'printf "%s\\n" "$TRIAL_ID" > "$LOGS/proxy_trial_id.txt"',
                '(sha256sum "$INSTRUCTION_FILE" 2>/dev/null || shasum -a 256 "$INSTRUCTION_FILE" 2>/dev/null || true) | cut -d" " -f1 > "$LOGS/instruction.sha256"',
                'case "$OPENAI_BASE_URL" in',
                '  */v1) export OPENAI_BASE_URL="${OPENAI_BASE_URL%/v1}/t/${TRIAL_ID}/v1" ;;',
                "esac",
            ]
        )

    exec_cmd = (
        "mizar exec --isolated --no-persist "
        "--provider openai "
        f"--protocol {protocol} "
        f"--model {shlex.quote(model)} "
        '--base-url "$OPENAI_BASE_URL" '
        f"--reasoning-effort {effort} "
        f"--max-tokens {max_tokens} "
    )
    if extra_args.strip():
        exec_cmd += f"{extra_args.strip()} "
    exec_cmd += '--json --prompt-file "$INSTRUCTION_FILE"'
    run_lines.append(exec_cmd)

    return {
        "import_path": "adapters.command_agent:CommandAgent",
        "model_name": f"openai/{model}",
        "override_setup_timeout_sec": 300,
        "env": {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "MIZAR_SKIP_VERSION_CHECK": "1",
        },
        "kwargs": {
            "harness": label,
            "version": version,
            "uploads": uploads,
            "install": install_lines,
            "files": {
                "$STATE/agent.env": [
                    "export MIZAR_SKIP_VERSION_CHECK='1'",
                    "export OPENAI_BASE_URL='${OPENAI_BASE_URL}'",
                    "export OPENAI_API_KEY='${OPENAI_API_KEY}'",
                ],
            },
            "run": run_lines,
        },
    }


def build_pi_agent(
    model: str,
    protocol: str,
    effort: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Build Harbor built-in Pi v1.4.0 config aligned on the same gateway."""
    model_api = "openai-responses" if protocol == "responses" else "openai-completions"
    return {
        "name": "pi",
        "model_name": f"openai/{model}",
        "override_setup_timeout_sec": 600,
        "env": {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "HARBOR_ALLOW_INSECURE_MODEL_BASE_URL": "true",
        },
        "kwargs": {
            "version": "1.4.0",
            "model_api": model_api,
            "thinking": effort,
        },
    }


def build_codex_agent(
    model: str,
    protocol: str,
    effort: str,
    base_url: str,
    api_key: str,
) -> dict[str, Any]:
    """Build Harbor built-in Codex CLI v0.164.0 config aligned on the same gateway."""
    wire_api = "responses" if protocol == "responses" else "chat"
    return {
        "name": "codex",
        "model_name": f"openai/{model}",
        "override_setup_timeout_sec": 900,
        "env": {
            "OPENAI_API_KEY": api_key,
            "OPENAI_BASE_URL": base_url,
            "HARBOR_ALLOW_INSECURE_MODEL_BASE_URL": "true",
        },
        "kwargs": {
            "version": "0.164.0",
            "reasoning_effort": effort,
            "config": {
                "model_provider": "study_proxy",
                "model_providers": {
                    "study_proxy": {
                        "name": "Terra study proxy",
                        "base_url": base_url,
                        "env_key": "OPENAI_API_KEY",
                        "wire_api": wire_api,
                        "supports_websockets": False,
                        "stream_idle_timeout_ms": 900000,
                        "stream_max_retries": 1,
                        "request_max_retries": 1,
                    }
                },
                "web_search": "disabled",
                "model_context_window": 1048576,
                "tool_output_token_limit": 10000,
            },
        },
    }


def build_job_config(args: argparse.Namespace, base_url: str, api_key: str) -> dict[str, Any]:
    """Assemble a complete Harbor JobConfig dict for Docker execution."""
    use_proxy = not args.no_study_proxy
    agents: list[dict[str, Any]] = []
    pi_protocol = args.protocol if args.pi_protocol == "same" else args.pi_protocol

    for name in dict.fromkeys(args.agents):
        if name == "mizar":
            label = args.mizar_label or "mizar"
            agents.append(
                build_mizar_agent(
                    model=args.model,
                    protocol=args.protocol,
                    effort=args.effort,
                    max_tokens=args.max_tokens,
                    base_url=agent_route_url(base_url, label, use_proxy),
                    api_key=api_key,
                    task_platform=args.task_platform,
                    label=label,
                    version=args.mizar_version,
                    mizar_bin=args.mizar_bin,
                    extra_args=args.mizar_args,
                    upload_rg=not args.no_upload_rg,
                    use_proxy=use_proxy,
                )
            )
        elif name == "pi":
            agents.append(
                build_pi_agent(
                    model=args.model,
                    protocol=pi_protocol,
                    effort=args.effort,
                    base_url=agent_route_url(base_url, "pi", use_proxy),
                    api_key=api_key,
                )
            )
        elif name == "codex":
            agents.append(
                build_codex_agent(
                    model=args.model,
                    protocol=args.protocol,
                    effort=args.effort,
                    base_url=agent_route_url(base_url, "codex", use_proxy),
                    api_key=api_key,
                )
            )

    dataset: dict[str, Any] = {
        "name": DATASET_REF,
        "version": DATASET_VERSION,
    }
    if args.tasks:
        dataset["task_names"] = list(args.tasks)

    return {
        "job_name": args.job_name,
        "jobs_dir": str(JOBS_DIR),
        "n_attempts": args.attempts,
        "n_concurrent_trials": args.concurrency,
        "retry": {
            "max_retries": args.max_retries,
        },
        "environment": {
            "type": "docker",
            "override_storage_mb": args.storage_mb,
        },
        "datasets": [dataset],
        "agents": agents,
    }


def build_docker_prefix(args: argparse.Namespace, container_name: str | None) -> list[str]:
    """Build the `docker run` command prefix sharing the workspace and Docker socket."""
    work = str(ROOT)
    cmd = ["docker", "run"]
    if args.detach:
        cmd.append("-d")
        if container_name:
            cmd.extend(["--name", container_name])
    else:
        cmd.append("--rm")
    if sys.platform.startswith("linux"):
        cmd.extend(["--add-host", "host.docker.internal:host-gateway"])
    cmd.extend(
        [
            "-v",
            "/var/run/docker.sock:/var/run/docker.sock",
            "-v",
            f"{work}:{work}",
            "-w",
            work,
            "-e",
            f"HOME={HARBOR_HOME}",
        ]
    )
    if args.task_platform != "auto":
        cmd.extend(["-e", f"DOCKER_DEFAULT_PLATFORM={args.task_platform}"])
    cmd.append(IMAGE_TAG)
    return cmd


def run_preflight(args: argparse.Namespace) -> None:
    """Verify Docker daemon and controller image before launching trials."""
    if args.dry_run or args.skip_preflight:
        return
    try:
        subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Docker daemon is not reachable: {exc}") from exc

    img_check = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if img_check.returncode != 0:
        raise RuntimeError(
            f"Missing Docker image {IMAGE_TAG}. Build it first with:\n"
            f"  docker build -t {IMAGE_TAG} - < Dockerfile"
        )


def _extract_proxy_port_from_job_config(cfg: dict[str, Any]) -> int | None:
    for agent in cfg.get("agents") or []:
        url = (agent.get("env") or {}).get("OPENAI_BASE_URL")
        if not url:
            continue
        parsed = urlsplit(url)
        if parsed.port and "/a/" in parsed.path:
            return int(parsed.port)
    return None


def _ensure_resume_proxy(args: argparse.Namespace, job_dir: Path) -> subprocess.Popen[str] | None:
    if args.no_study_proxy:
        return None
    meta_path = JOBS_DIR / f"{job_dir.name}.meta.json"
    meta: dict[str, Any] = {}
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    job_cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))

    port = meta.get("proxy_port") or _extract_proxy_port_from_job_config(job_cfg)
    if not port or is_port_listening("127.0.0.1", int(port)):
        return None

    upstream = args.base_url or os.environ.get("OPENAI_BASE_URL") or meta.get("upstream_url")
    if not upstream:
        raise RuntimeError(
            f"Study proxy on port {port} is stopped and upstream URL is unknown; pass --base-url to resume."
        )
    effort = meta.get("effort") or args.effort
    max_tokens = int(meta.get("max_tokens") or args.max_tokens)
    force_model = meta.get("upstream_model") or meta.get("model") or args.model
    usage_log = Path(meta.get("usage_log_path") or (JOBS_DIR / f"{job_dir.name}.proxy_usage.jsonl"))
    print(f"Restarting stopped study proxy on port {port} -> {upstream}", flush=True)
    proc, _ = start_study_proxy(
        upstream_url=upstream,
        effort=effort,
        max_tokens=max_tokens,
        force_model=force_model,
        port=int(port),
        job_name=job_dir.name,
        detach=args.detach,
        usage_log_path=usage_log,
    )
    return proc


def run_resume(args: argparse.Namespace) -> int:
    """Resume an unfinished Harbor job or clear matching infra errors and retry."""
    job_path = Path(args.resume)
    if not job_path.is_absolute():
        candidate = JOBS_DIR / args.resume
        job_path = candidate if candidate.exists() else (ROOT / args.resume)
    job_path = job_path.resolve()
    if not (job_path / "config.json").is_file():
        raise FileNotFoundError(f"Missing Harbor job config.json under {job_path}")

    run_preflight(args)
    proxy_proc = _ensure_resume_proxy(args, job_path)
    try:
        container_name = f"harbor-resume-{job_path.name}"
        cmd = build_docker_prefix(args, container_name=container_name)
        cmd.extend(["jobs", "resume", "-p", str(job_path)])
        for err_type in args.filter_error_type:
            cmd.extend(["-f", err_type])

        print(f"Resuming Harbor job: {job_path}", flush=True)
        return launch_container(cmd, detach=args.detach, container_name=container_name, job_dir=job_path)
    finally:
        if proxy_proc is not None and not args.detach:
            _stop_proxy(proxy_proc)


def _stop_proxy(proc: subprocess.Popen[str]) -> None:
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


def launch_container(
    cmd: list[str],
    detach: bool,
    container_name: str,
    job_dir: Path,
) -> int:
    """Run Harbor synchronously or detached inside the Docker daemon."""
    if not detach:
        return subprocess.call(cmd)

    # Remove an exited container of the same name so re-runs stay idempotent.
    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True)
    cid = subprocess.check_output(cmd, text=True).strip()
    print(f"Started detached Harbor controller container: {container_name} ({cid[:12]})")
    print(f"Follow logs : docker logs -f {container_name}")
    print(f"Score live  : python3 score_report.py {job_dir}")
    pid_file = JOBS_DIR / f"{job_dir.name}.proxy.pid"
    if pid_file.is_file():
        print(f"Stop proxy  : kill $(cat {pid_file})  (after container finishes)")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agents",
        nargs="+",
        choices=("mizar", "pi", "codex"),
        default=["mizar"],
        help="Harnesses to evaluate (default: mizar)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model ID passed to harnesses (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--upstream-model",
        help="Optional upstream model ID rewrite applied by eval_proxy.py (defaults to --model)",
    )
    parser.add_argument(
        "--protocol",
        choices=("responses", "completions"),
        default="responses",
        help="Model wire protocol (default: responses)",
    )
    parser.add_argument(
        "--pi-protocol",
        choices=("same", "responses", "completions"),
        default="same",
        help="Optional override for Pi wire API (default: follow --protocol)",
    )
    parser.add_argument(
        "--effort",
        choices=EFFORT_CHOICES,
        default=DEFAULT_EFFORT,
        help=f"Target reasoning effort enforced via study proxy (default: {DEFAULT_EFFORT})",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--no-study-proxy",
        action="store_true",
        help="Disable eval_proxy.py enforcement/telemetry and talk directly to upstream base URL",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        default=0,
        help="Port for study proxy on host (default: 3426 or next free port)",
    )
    parser.add_argument(
        "--proxy-host",
        default="auto",
        help="Host IP/hostname containers use to reach study proxy (default: auto; bridge gateway on Linux)",
    )
    parser.add_argument("--mizar-bin", help="Custom Mizar Linux binary file or arch dir (overrides vendor/bin)")
    parser.add_argument("--mizar-label", default="mizar", help="Harness label for Mizar in reports (default: mizar)")
    parser.add_argument("--mizar-version", default="0.1.5", help="Mizar version tag recorded in config (default: 0.1.5)")
    parser.add_argument("--mizar-args", default="", help="Extra CLI flags appended to `mizar exec`")
    parser.add_argument(
        "--no-upload-rg",
        action="store_true",
        help="Do not upload ripgrep into Mizar containers (strict container tool parity with Pi/Codex)",
    )
    parser.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN, help="USD per 1M uncached input tokens")
    parser.add_argument("--price-cached", type=float, default=DEFAULT_PRICE_CACHED, help="USD per 1M cached input tokens")
    parser.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT, help="USD per 1M output tokens")
    parser.add_argument(
        "--price-cache-write",
        type=float,
        default=DEFAULT_PRICE_CACHE_WRITE,
        help="USD per 1M cache-write tokens",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=[],
        help='Optional task filter patterns, e.g. "*/fix-git" (omit for all 62 tasks)',
    )
    parser.add_argument("--attempts", type=int, default=1, help="Attempts per task and harness (default: 1)")
    parser.add_argument("--concurrency", type=int, default=2, help="Concurrent trials (default: 2)")
    parser.add_argument("--max-retries", type=int, default=2, help="Harbor transient infra retry count (default: 2)")
    parser.add_argument("--storage-mb", type=int, default=32768, help="Container storage override MB (default: 32768)")
    parser.add_argument(
        "--task-platform",
        default="linux/amd64",
        help="DOCKER_DEFAULT_PLATFORM for task containers (default: linux/amd64; use 'auto' for host default)",
    )
    parser.add_argument("--base-url", help="Upstream OpenAI-compatible base URL (or set OPENAI_BASE_URL)")
    parser.add_argument("--api-key", help="Upstream API key (or set OPENAI_API_KEY)")
    parser.add_argument("--job-name", help="Custom Harbor job name")
    parser.add_argument("--detach", action="store_true", help="Run Harbor controller detached in Docker daemon")
    parser.add_argument("--resume", help="Resume an existing job by name or path under jobs/")
    parser.add_argument(
        "--filter-error-type",
        nargs="+",
        default=[],
        help="With --resume, clear trials matching these exception types before resuming",
    )
    parser.add_argument("--skip-preflight", action="store_true", help="Skip Docker daemon and image existence checks")
    parser.add_argument("--dry-run", action="store_true", help="Validate Harbor config without running trials")
    return parser.parse_args(argv)


def _write_secure_json(path: Path, payload: dict[str, Any], mode: int = 0o600) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.resume:
        return run_resume(args)

    upstream_url = args.base_url or os.environ.get("OPENAI_BASE_URL")
    if not upstream_url:
        raise ValueError("Missing upstream API URL: pass --base-url or set OPENAI_BASE_URL")
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Missing upstream API key: pass --api-key or set OPENAI_API_KEY")

    if not args.job_name:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        scope = "full" if not args.tasks else "subset"
        labels = [args.mizar_label if a == "mizar" else a for a in dict.fromkeys(args.agents)]
        args.job_name = f"{'-'.join(labels)}-terra-{args.effort}-{scope}-{stamp}"

    run_preflight(args)
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    HARBOR_HOME.mkdir(parents=True, exist_ok=True)

    use_proxy = not args.no_study_proxy
    usage_log_path = JOBS_DIR / f"{args.job_name}.proxy_usage.jsonl"
    proxy_proc: subprocess.Popen[str] | None = None
    proxy_port: int | None = None
    proxy_host = resolve_proxy_host(args.proxy_host) if use_proxy else None
    force_model = args.upstream_model or args.model

    if use_proxy and not args.dry_run:
        port = args.proxy_port or pick_free_port(3426)
        proxy_proc, proxy_port = start_study_proxy(
            upstream_url=upstream_url,
            effort=args.effort,
            max_tokens=args.max_tokens,
            force_model=force_model,
            port=port,
            job_name=args.job_name,
            detach=args.detach,
            usage_log_path=usage_log_path,
        )
        base_url = f"http://{proxy_host}:{proxy_port}/v1"
    elif use_proxy and args.dry_run:
        proxy_port = args.proxy_port or 3426
        base_url = f"http://{proxy_host}:{proxy_port}/v1"
    else:
        base_url = upstream_url

    try:
        config = build_job_config(args, base_url=base_url, api_key=api_key)
        config_path = JOBS_DIR / f"{args.job_name}.config.json"
        _write_secure_json(config_path, config, mode=0o600)

        meta = {
            "job_name": args.job_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "upstream_model": force_model,
            "protocol": args.protocol,
            "pi_protocol": args.protocol if args.pi_protocol == "same" else args.pi_protocol,
            "effort": args.effort,
            "max_tokens": args.max_tokens,
            "upstream_url": upstream_url,
            "use_proxy": use_proxy,
            "proxy_host": proxy_host,
            "proxy_port": proxy_port,
            "usage_log_path": str(usage_log_path) if use_proxy else None,
            "agents": [args.mizar_label if a == "mizar" else a for a in dict.fromkeys(args.agents)],
            "mizar_version": args.mizar_version,
            "rg_uploaded": (not args.no_upload_rg) if "mizar" in args.agents else None,
            "pricing_per_1m_usd": {
                "uncached_input": args.price_in,
                "cached_input": args.price_cached,
                "cache_write": args.price_cache_write,
                "output": args.price_out,
            },
        }
        _write_secure_json(JOBS_DIR / f"{args.job_name}.meta.json", meta, mode=0o644)

        proxy_note = f"study_proxy({proxy_host}:{proxy_port})->{upstream_url}" if use_proxy else f"direct({upstream_url})"
        print(
            f"Job: {args.job_name} | agents={','.join(meta['agents'])} | model={args.model} "
            f"| protocol={args.protocol} | effort={args.effort} | concurrency={args.concurrency} "
            f"| {proxy_note}",
            flush=True,
        )

        container_name = f"harbor-{args.job_name}"
        cmd = build_docker_prefix(args, container_name=container_name)
        cmd.extend(["run", "-c", str(config_path)])
        if args.dry_run:
            cmd.append("--dry-run")

        code = launch_container(
            cmd,
            detach=args.detach and not args.dry_run,
            container_name=container_name,
            job_dir=JOBS_DIR / args.job_name,
        )
        if not args.dry_run and not args.detach:
            print(f"\nRun `python3 score_report.py {JOBS_DIR / args.job_name}` to generate full report.", flush=True)
        return code
    finally:
        if proxy_proc is not None and not args.detach:
            _stop_proxy(proxy_proc)


if __name__ == "__main__":
    raise SystemExit(main())
