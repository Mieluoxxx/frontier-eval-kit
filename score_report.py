#!/usr/bin/env python3
"""Score and compare FrontierHarness Eval v1.1.0 trials for Mizar, Pi, and Codex."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import html
import json
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
JOBS_DIR = ROOT / "jobs"
IMAGES_MANIFEST = ROOT / "frontier-v1.1-images.json"
TASKS_DOC = ROOT / "题目.txt"

DEFAULT_PRICING = {
    "uncached_input": 1.00,
    "cached_input": 0.05,
    "cache_write": 0.0,
    "output": 5.00,
}

CATEGORY_ALIASES = {
    "software": "Software Engineering",
    "software-engineering": "Software Engineering",
    "science": "Science & Computing",
    "scientific-computing": "Science & Computing",
    "mathematics": "Science & Computing",
    "ml": "ML & Training",
    "machine-learning": "ML & Training",
    "model-training": "ML & Training",
    "security": "Security",
    "system-administration": "Systems & Ops",
    "operations": "Systems & Ops",
    "data-processing": "Data & Querying",
    "data-querying": "Data & Querying",
    "data-science": "Data & Querying",
    "debugging": "Debugging",
}

SELF_CHECK_CMD_RE = re.compile(
    r"\b(pytest|unittest|cargo\s+test|go\s+test|npm\s+test|bun\s+test|make\s+(?:-\S+\s+)*test|py_compile|sparql|curl\s|verify)\b",
    re.I,
)


def parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    dt = datetime.fromisoformat(normalized)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def span_seconds(block: dict[str, Any] | None) -> float | None:
    if not block:
        return None
    start = parse_iso(block.get("started_at"))
    end = parse_iso(block.get("finished_at"))
    if start and end:
        return max(0.0, (end - start).total_seconds())
    return None


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * pct
    lo = int(pos)
    hi = min(len(ordered) - 1, lo + 1)
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def normalize_model_name(name: str | None) -> str:
    if not name:
        return "unknown"
    return name[len("openai/") :] if name.startswith("openai/") else name


def classify_trial(result: dict[str, Any]) -> tuple[str, float | None, str | None]:
    """Return (outcome, reward, error_type) aligned with FrontierHarness v1.1 rules."""
    exc = result.get("exception_info")
    exc_type = exc.get("exception_type") if isinstance(exc, dict) else None
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    numeric_rewards = [float(v) for v in rewards.values() if isinstance(v, (int, float))]
    reward = (
        float(rewards["reward"])
        if isinstance(rewards.get("reward"), (int, float))
        else (max(numeric_rewards) if numeric_rewards else None)
    )

    if reward is not None and reward >= 1.0:
        return "pass", reward, exc_type
    if exc_type == "AgentTimeoutError":
        return "timeout", reward if reward is not None else 0.0, exc_type
    if exc_type:
        return "infra", reward, exc_type
    if result.get("finished_at") is None:
        return "running", reward, None
    return "fail", reward if reward is not None else 0.0, None


def load_task_categories(doc_path: Path = TASKS_DOC) -> dict[str, str]:
    """Parse task -> normalized category from 题目.txt when present."""
    if not doc_path.is_file():
        return {}
    mapping: dict[str, str] = {}
    current_task: str | None = None
    for line in doc_path.read_text(encoding="utf-8").splitlines():
        m_task = re.match(r"^\[\d+\]\s+(terminal-bench/\S+)", line.strip())
        if m_task:
            current_task = m_task.group(1)
            continue
        if current_task:
            m_cat = re.search(r"分类：([^|]+)", line)
            if m_cat:
                raw = m_cat.group(1).strip().lower()
                mapping[current_task] = CATEGORY_ALIASES.get(raw, "Other")
                current_task = None
    return mapping


def load_manifest_tasks(manifest_path: Path = IMAGES_MANIFEST) -> list[dict[str, Any]]:
    if not manifest_path.is_file():
        return []
    categories = load_task_categories()
    tasks = json.loads(manifest_path.read_text(encoding="utf-8")).get("tasks", [])
    for task in tasks:
        tname = task.get("task", "")
        task["subset"] = "tb4" if task.get("separate_verifier_image") else "tb21"
        task["category"] = categories.get(tname, "Other")
    return tasks


def load_job_meta(job_dir: Path) -> dict[str, Any]:
    for candidate in (JOBS_DIR / f"{job_dir.name}.meta.json", job_dir / "meta.json"):
        if candidate.is_file():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
    return {}


def resolve_pricing(args: argparse.Namespace, metas: list[dict[str, Any]]) -> tuple[dict[str, float], str]:
    base = dict(DEFAULT_PRICING)
    source = "terra_default"
    for meta in metas:
        p = meta.get("pricing_per_1m_usd")
        if isinstance(p, dict):
            for k in base:
                if isinstance(p.get(k), (int, float)):
                    base[k] = float(p[k])
            source = "job_meta"
            break
    overrides = {
        "uncached_input": args.price_in,
        "cached_input": args.price_cached,
        "cache_write": args.price_cache_write,
        "output": args.price_out,
    }
    if any(v is not None for v in overrides.values()):
        for k, v in overrides.items():
            if v is not None:
                base[k] = float(v)
        source = "cli_override"
    return base, source


def compute_call_cost(rec: dict[str, Any], pricing: dict[str, float]) -> float:
    upstream = rec.get("upstream_cost_usd")
    if isinstance(upstream, (int, float)):
        return float(upstream)
    uncached = int(rec.get("uncached_input_tokens") or 0)
    cached = int(rec.get("cached_tokens") or 0)
    cwrite = int(rec.get("cache_write_tokens") or 0)
    output = int(rec.get("output_tokens") or 0)
    return (
        uncached * pricing["uncached_input"]
        + cached * pricing["cached_input"]
        + cwrite * pricing["cache_write"]
        + output * pricing["output"]
    ) / 1_000_000.0


def load_proxy_usage(job_dir: Path, meta: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    if meta.get("usage_log_path"):
        candidates.append(Path(meta["usage_log_path"]))
    candidates.append(JOBS_DIR / f"{job_dir.name}.proxy_usage.jsonl")
    candidates.append(job_dir / "proxy_usage.jsonl")

    seen: set[Path] = set()
    rows: list[dict[str, Any]] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            continue
        seen.add(resolved)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                item["_ts_dt"] = parse_iso(item.get("ts"))
                rows.append(item)
        break
    rows.sort(key=lambda r: r.get("ts") or "")
    return rows


def _inspect_trial_agent_artifacts(agent_dir: Path) -> dict[str, Any]:
    """Extract behavioral, token fallback, and attribution metadata from trial agent/ dir."""
    info: dict[str, Any] = {
        "proxy_trial_id": None,
        "session_id": None,
        "instruction_sha256": None,
        "mizar_stdout": None,
        "trajectory_metrics": None,
    }
    if not agent_dir.is_dir():
        return info

    tid_file = agent_dir / "proxy_trial_id.txt"
    if tid_file.is_file():
        val = tid_file.read_text(encoding="utf-8", errors="replace").strip()
        if val:
            info["proxy_trial_id"] = val

    sha_file = agent_dir / "instruction.sha256"
    if sha_file.is_file():
        val = sha_file.read_text(encoding="utf-8", errors="replace").strip()
        if val:
            info["instruction_sha256"] = val[:16]

    stdout_file = agent_dir / "stdout.log"
    if stdout_file.is_file() and stdout_file.stat().st_size < 262144:
        for line in reversed(stdout_file.read_text(encoding="utf-8", errors="replace").splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                try:
                    parsed = json.loads(line)
                    if isinstance(parsed, dict) and ("steps" in parsed or "session_id" in parsed):
                        info["mizar_stdout"] = parsed
                        info["session_id"] = parsed.get("session_id")
                        break
                except json.JSONDecodeError:
                    pass

    traj_file = agent_dir / "trajectory.json"
    if traj_file.is_file():
        try:
            traj = json.loads(traj_file.read_text(encoding="utf-8", errors="replace"))
        except json.JSONDecodeError:
            traj = None
        if isinstance(traj, dict):
            if traj.get("session_id"):
                info["session_id"] = str(traj["session_id"])
            steps = traj.get("steps") or []
            agent_turns = 0
            turns_with_tools = 0
            tool_calls = 0
            tool_errors = 0
            self_checked = False
            consecutive_err_streak = 0
            doom_loop = False
            step_usage_calls: list[dict[str, Any]] = []

            for st in steps:
                if not isinstance(st, dict):
                    continue
                if st.get("source") == "user" and not info["instruction_sha256"]:
                    msg = st.get("message")
                    if isinstance(msg, str) and not msg.lstrip().startswith("<environment_context>"):
                        info["instruction_sha256"] = hashlib.sha256(msg.strip().encode("utf-8")).hexdigest()[:16]
                if st.get("source") != "agent":
                    continue
                agent_turns += 1
                tcs = st.get("tool_calls") or []
                if tcs:
                    turns_with_tools += 1
                    tool_calls += len(tcs)
                    for tc in tcs:
                        if not isinstance(tc, dict):
                            continue
                        args_str = json.dumps(tc.get("arguments") or {}, ensure_ascii=False)
                        if SELF_CHECK_CMD_RE.search(args_str):
                            self_checked = True
                obs_results = (st.get("observation") or {}).get("results") or []
                turn_had_err = False
                for res in obs_results:
                    if not isinstance(res, dict):
                        continue
                    extra = res.get("extra") or {}
                    is_err = bool(extra.get("is_error"))
                    content = res.get("content")
                    if not is_err and isinstance(content, str) and "\nProcess exited with code " in content:
                        if "Process exited with code 0" not in content:
                            is_err = True
                    if is_err:
                        tool_errors += 1
                        turn_had_err = True
                if tcs and turn_had_err:
                    consecutive_err_streak += 1
                    if consecutive_err_streak >= 3:
                        doom_loop = True
                elif tcs:
                    consecutive_err_streak = 0

                metrics = st.get("metrics")
                if isinstance(metrics, dict) and ("prompt_tokens" in metrics or "completion_tokens" in metrics):
                    raw_in = int(metrics.get("prompt_tokens") or 0)
                    cached = int(metrics.get("cached_tokens") or 0)
                    step_extra = metrics.get("extra") or {}
                    step_usage_calls.append(
                        {
                            "ts": st.get("timestamp"),
                            "input_tokens": raw_in,
                            "cached_tokens": cached,
                            "uncached_input_tokens": max(0, raw_in - cached),
                            "cache_write_tokens": int(step_extra.get("cache_write_tokens") or 0),
                            "output_tokens": int(metrics.get("completion_tokens") or 0),
                            "reasoning_tokens": int(step_extra.get("reasoning_tokens") or 0),
                            "tool_calls_emitted": len(tcs),
                        }
                    )

            info["trajectory_metrics"] = {
                "agent_turns": agent_turns,
                "turns_with_tools": turns_with_tools,
                "tool_calls": tool_calls,
                "tool_errors": tool_errors,
                "self_checked": self_checked,
                "doom_loop": doom_loop,
                "step_calls": step_usage_calls,
            }
    return info


def collect_job_trials(job_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load job config and scan all trial directories (works mid-run and post-run)."""
    job_cfg = json.loads((job_dir / "config.json").read_text(encoding="utf-8"))
    trials: list[dict[str, Any]] = []

    for sub in sorted(job_dir.iterdir()):
        if not sub.is_dir():
            continue
        res_path = sub / "result.json"
        cfg_path = sub / "config.json"
        if not res_path.is_file() and not cfg_path.is_file():
            continue

        if res_path.is_file():
            res = json.loads(res_path.read_text(encoding="utf-8"))
            outcome, reward, err_type = classify_trial(res)
            task_name = res.get("task_name") or sub.name.split("__")[0]
            agent_info = res.get("agent_info") or {}
            harness = agent_info.get("name") or "unknown"
            model = normalize_model_name((agent_info.get("model_info") or {}).get("name"))
            agent_sec = span_seconds(res.get("agent_execution"))
            verifier_sec = span_seconds(res.get("verifier"))
            setup_sec = (span_seconds(res.get("environment_setup")) or 0.0) + (
                span_seconds(res.get("agent_setup")) or 0.0
            )
            run_sec = (
                (agent_sec or 0.0) + (verifier_sec or 0.0)
                if (agent_sec is not None or verifier_sec is not None)
                else None
            )
            wall_sec = span_seconds({"started_at": res.get("started_at"), "finished_at": res.get("finished_at")})
            exc_msg = (
                ((res.get("exception_info") or {}).get("exception_message") or "").splitlines()[0]
                if res.get("exception_info")
                else None
            )
            agent_execution = res.get("agent_execution")
            agent_result = res.get("agent_result") or {}
        else:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            task_name = (cfg.get("task") or {}).get("name") or sub.name.split("__")[0]
            agent_cfg = cfg.get("agent") or {}
            harness = (agent_cfg.get("kwargs") or {}).get("harness") or agent_cfg.get("name") or "unknown"
            model = normalize_model_name(agent_cfg.get("model_name"))
            outcome, reward, err_type, exc_msg = "running", None, None, None
            agent_sec = verifier_sec = run_sec = wall_sec = None
            setup_sec = None
            agent_execution = None
            agent_result = {}

        if not task_name.startswith("terminal-bench/") and "/" not in task_name:
            task_name = f"terminal-bench/{task_name}"

        agent_artifacts = _inspect_trial_agent_artifacts(sub / "agent")
        trials.append(
            {
                "job_name": job_dir.name,
                "trial_dir": sub.name,
                "task_name": task_name,
                "harness": harness,
                "raw_harness": harness,
                "model": model,
                "outcome": outcome,
                "reward": reward,
                "error_type": err_type,
                "error_message": exc_msg,
                "agent_sec": agent_sec,
                "verifier_sec": verifier_sec,
                "setup_sec": setup_sec,
                "run_sec": run_sec,
                "wall_sec": wall_sec,
                "agent_execution": agent_execution,
                "agent_result": agent_result,
                "artifacts": agent_artifacts,
                "proxy_calls": [],
            }
        )
    return job_cfg, trials


def attribute_proxy_calls(
    trials: list[dict[str, Any]],
    proxy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach proxy calls to exact trials by trial_id, session_id, response chain, prompt hash, or time window."""
    by_proxy_trial_id: dict[str, int] = {}
    by_session_id: dict[str, list[int]] = defaultdict(list)
    by_prompt_harness: dict[tuple[str, str], list[int]] = defaultdict(list)
    windows: list[tuple[int, str, datetime, datetime]] = []

    for idx, tr in enumerate(trials):
        art = tr["artifacts"]
        if art.get("proxy_trial_id"):
            by_proxy_trial_id[art["proxy_trial_id"]] = idx
        if art.get("session_id"):
            by_session_id[art["session_id"]].append(idx)
        if art.get("instruction_sha256"):
            by_prompt_harness[(art["instruction_sha256"], tr["harness"])].append(idx)
        start = parse_iso((tr.get("agent_execution") or {}).get("started_at"))
        end = parse_iso((tr.get("agent_execution") or {}).get("finished_at"))
        if start and end:
            windows.append((idx, tr["harness"], start, end))

    response_to_trial: dict[str, int] = {}
    unassigned_rows: list[dict[str, Any]] = []

    for row in proxy_rows:
        matched_idx: int | None = None
        tid = row.get("trial_id")
        if tid and tid in by_proxy_trial_id:
            matched_idx = by_proxy_trial_id[tid]

        if matched_idx is None and row.get("prompt_cache_key"):
            cands = by_session_id.get(str(row["prompt_cache_key"]), [])
            if len(cands) == 1:
                matched_idx = cands[0]

        if matched_idx is None and row.get("previous_response_id"):
            matched_idx = response_to_trial.get(str(row["previous_response_id"]))

        ts = row.get("_ts_dt")
        row_harness = row.get("harness")

        if matched_idx is None and row.get("user_prompt_sha256"):
            key_cands: list[int] = []
            for h_key in ([row_harness] if row_harness else list({t["harness"] for t in trials})):
                key_cands.extend(by_prompt_harness.get((row["user_prompt_sha256"], h_key), []))
            if len(key_cands) == 1:
                matched_idx = key_cands[0]
            elif len(key_cands) > 1 and ts is not None:
                in_win = [
                    i
                    for i, _h, st, en in windows
                    if i in key_cands and (st.timestamp() - 10.0) <= ts.timestamp() <= (en.timestamp() + 10.0)
                ]
                if len(in_win) == 1:
                    matched_idx = in_win[0]

        if matched_idx is None and ts is not None:
            in_win = [
                i
                for i, h, st, en in windows
                if (row_harness is None or h == row_harness)
                and (st.timestamp() - 5.0) <= ts.timestamp() <= (en.timestamp() + 5.0)
            ]
            if len(in_win) == 1:
                matched_idx = in_win[0]

        if matched_idx is not None:
            trials[matched_idx]["proxy_calls"].append(row)
            if row.get("response_id"):
                response_to_trial[str(row["response_id"])] = matched_idx
        else:
            unassigned_rows.append(row)
    return unassigned_rows


def _compute_cache_and_usage_stats(
    calls: list[dict[str, Any]],
    pricing: dict[str, float],
) -> dict[str, Any]:
    """Compute cache hit, warmup hit, breaks, tokens, TTFT, and error stats for a call list."""
    if not calls:
        return {
            "calls": 0,
            "main_calls": 0,
            "cost_usd": None,
            "uncached_input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "total_input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
            "cache_hit_rate": None,
            "warmup_cache_hit_rate": None,
            "warmup_input_tokens": 0,
            "warmup_cached_tokens": 0,
            "key_coverage": None,
            "cache_breaks": 0,
            "peak_context_tokens": None,
            "avg_ttft_ms": None,
            "p95_ttft_ms": None,
            "llm_api_sec": 0.0,
            "http_errors": 0,
            "rate_limits": 0,
            "efforts": {},
            "tool_calls_emitted": 0,
            "turns_with_tools": 0,
        }

    total_cost = 0.0
    total_uncached = 0
    total_cached = 0
    total_cwrite = 0
    total_in = 0
    total_out = 0
    total_reason = 0
    http_errors = 0
    rate_limits = 0
    efforts: Counter[str] = Counter()
    ttfts: list[float] = []
    api_ms = 0
    tool_calls_emitted = 0
    turns_with_tools = 0

    main_calls = [c for c in calls if not c.get("is_side_call")]
    for c in calls:
        total_cost += compute_call_cost(c, pricing)
        total_uncached += int(c.get("uncached_input_tokens") or 0)
        total_cached += int(c.get("cached_tokens") or 0)
        total_cwrite += int(c.get("cache_write_tokens") or 0)
        total_in += int(c.get("input_tokens") or 0)
        total_out += int(c.get("output_tokens") or 0)
        total_reason += int(c.get("reasoning_tokens") or 0)
        status = int(c.get("status") or 200)
        if status >= 400 or c.get("error"):
            http_errors += 1
        if status == 429:
            rate_limits += 1
        if c.get("effort"):
            efforts[str(c["effort"])] += 1
        if isinstance(c.get("ttft_ms"), (int, float)):
            ttfts.append(float(c["ttft_ms"]))
        if isinstance(c.get("duration_ms"), (int, float)):
            api_ms += int(c["duration_ms"])
        tc = int(c.get("tool_calls_emitted") or 0)
        tool_calls_emitted += tc
        if tc > 0:
            turns_with_tools += 1

    keys_present = sum(1 for c in main_calls if c.get("prompt_cache_key"))
    warmup_calls = main_calls[1:] if len(main_calls) > 1 else []
    warmup_in = sum(int(c.get("input_tokens") or 0) for c in warmup_calls)
    warmup_cached = sum(int(c.get("cached_tokens") or 0) for c in warmup_calls)

    cache_breaks = 0
    prev_cached = 0
    for c in main_calls:
        curr_cached = int(c.get("cached_tokens") or 0)
        if prev_cached > 0 and (curr_cached == 0 or curr_cached < prev_cached * 0.8):
            cache_breaks += 1
        prev_cached = curr_cached

    in_vals = [int(c.get("input_tokens") or 0) for c in main_calls if int(c.get("input_tokens") or 0) > 0]
    denom = total_uncached + total_cached
    return {
        "calls": len(calls),
        "main_calls": len(main_calls),
        "cost_usd": total_cost,
        "uncached_input_tokens": total_uncached,
        "cache_read_tokens": total_cached,
        "cache_write_tokens": total_cwrite,
        "total_input_tokens": total_in,
        "output_tokens": total_out,
        "reasoning_tokens": total_reason,
        "cache_hit_rate": (total_cached / denom) if denom else None,
        "warmup_cache_hit_rate": (warmup_cached / warmup_in) if warmup_in else None,
        "warmup_input_tokens": warmup_in,
        "warmup_cached_tokens": warmup_cached,
        "key_coverage": (keys_present / len(main_calls)) if main_calls else None,
        "cache_breaks": cache_breaks,
        "peak_context_tokens": max(in_vals) if in_vals else None,
        "avg_ttft_ms": (sum(ttfts) / len(ttfts)) if ttfts else None,
        "p95_ttft_ms": percentile(ttfts, 0.95),
        "llm_api_sec": api_ms / 1000.0,
        "http_errors": http_errors,
        "rate_limits": rate_limits,
        "efforts": dict(efforts),
        "tool_calls_emitted": tool_calls_emitted,
        "turns_with_tools": turns_with_tools,
    }


def _enrich_trial_telemetry(
    tr: dict[str, Any],
    pricing: dict[str, float],
    has_proxy_log: bool,
) -> None:
    """Attach unified usage & behavioral metrics onto each trial dict."""
    proxy_calls = tr["proxy_calls"]
    art = tr["artifacts"]
    mizar_out = art.get("mizar_stdout") or {}
    traj = art.get("trajectory_metrics") or {}

    if proxy_calls:
        u = _compute_cache_and_usage_stats(proxy_calls, pricing)
        u["source"] = "proxy_jsonl"
        u["distinct_keys"] = len({c["prompt_cache_key"] for c in proxy_calls if c.get("prompt_cache_key")})
        main_calls = [c for c in proxy_calls if not c.get("is_side_call")]
        u["context_growth_per_turn"] = (
            (int(main_calls[-1].get("input_tokens") or 0) - int(main_calls[0].get("input_tokens") or 0))
            / (len(main_calls) - 1)
            if len(main_calls) >= 2
            else None
        )
    elif not has_proxy_log and traj.get("step_calls"):
        u = _compute_cache_and_usage_stats(traj["step_calls"], pricing)
        u["source"] = "trajectory_fallback"
        u["distinct_keys"] = None
        u["key_coverage"] = None
        u["context_growth_per_turn"] = None
    elif not has_proxy_log and tr["agent_result"].get("n_input_tokens") is not None:
        raw_in = int(tr["agent_result"].get("n_input_tokens") or 0)
        cached = int(tr["agent_result"].get("n_cache_tokens") or 0)
        out_tok = int(tr["agent_result"].get("n_output_tokens") or 0)
        synthetic = [
            {
                "input_tokens": raw_in,
                "cached_tokens": cached,
                "uncached_input_tokens": max(0, raw_in - cached),
                "cache_write_tokens": 0,
                "output_tokens": out_tok,
                "reasoning_tokens": 0,
            }
        ]
        u = _compute_cache_and_usage_stats(synthetic, pricing)
        u["source"] = "agent_result_fallback"
        u["calls"] = 0
        u["main_calls"] = 0
        u["warmup_cache_hit_rate"] = None
        u["key_coverage"] = None
        u["distinct_keys"] = None
        u["cache_breaks"] = 0
        u["context_growth_per_turn"] = None
    else:
        u = None

    turns = (
        (u["main_calls"] if u and u["main_calls"] else None)
        or mizar_out.get("steps")
        or traj.get("agent_turns")
    )
    tool_calls = (
        traj.get("tool_calls")
        if traj
        else (mizar_out.get("tool_calls") if mizar_out.get("tool_calls") is not None else (u["tool_calls_emitted"] if u else None))
    )
    llm_sec = (
        (mizar_out["llm_ms"] / 1000.0)
        if isinstance(mizar_out.get("llm_ms"), (int, float))
        else (u["llm_api_sec"] if u and u["llm_api_sec"] > 0 else None)
    )
    tr["usage"] = u
    tr["behavior"] = {
        "turns": turns,
        "tool_calls": tool_calls,
        "turns_with_tools": traj.get("turns_with_tools") or (u["turns_with_tools"] if u else None),
        "tool_errors": traj.get("tool_errors") if traj else None,
        "self_checked": traj.get("self_checked") if traj else None,
        "doom_loop": traj.get("doom_loop") if traj else None,
        "stop_reason": mizar_out.get("reason"),
        "llm_sec": llm_sec,
    }


def _resolve_planned_tasks(
    job_cfgs: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    manifest_tasks: list[dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    manifest_map = {t["task"]: t for t in manifest_tasks}
    if not manifest_tasks:
        names = sorted({t["task_name"] for t in trials})
        return names, {n: {"task": n, "subset": "unknown", "category": "Other"} for n in names}

    patterns: list[str] = []
    unfiltered_job = False
    for cfg in job_cfgs:
        job_pats = [pat for d in (cfg.get("datasets") or []) for pat in (d.get("task_names") or [])]
        if not job_pats:
            unfiltered_job = True
        else:
            patterns.extend(job_pats)

    if unfiltered_job or not patterns:
        return [t["task"] for t in manifest_tasks], manifest_map

    matched: list[str] = []
    for t in manifest_tasks:
        tname = t["task"]
        short = tname.split("/", 1)[-1]
        if any(fnmatch.fnmatch(tname, p) or fnmatch.fnmatch(short, p) for p in patterns):
            matched.append(tname)
    if not matched:
        matched = sorted({t["task_name"] for t in trials})
    return matched, manifest_map


def _aggregate_harness_stats(
    harness: str,
    model: str,
    h_trials: list[dict[str, Any]],
    unassigned_calls: list[dict[str, Any]],
    pricing: dict[str, float],
    n_attempts: int,
    manifest_map: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    scored = [t for t in h_trials if t["outcome"] in {"pass", "fail", "timeout"}]
    passes = [t for t in scored if t["outcome"] == "pass"]
    fails = [t for t in scored if t["outcome"] == "fail"]
    timeouts = [t for t in scored if t["outcome"] == "timeout"]
    infra = [t for t in h_trials if t["outcome"] == "infra"]
    running = [t for t in h_trials if t["outcome"] == "running"]

    run_secs = [t["run_sec"] for t in scored if t["run_sec"] is not None]
    exec_secs = [t["agent_sec"] for t in scored if t["agent_sec"] is not None]

    # Collect per-harness call stream (assigned trial calls + unassigned same-harness calls)
    all_proxy_calls = [c for t in h_trials for c in t["proxy_calls"]] + list(unassigned_calls)
    if all_proxy_calls:
        usage_agg = _compute_cache_and_usage_stats(all_proxy_calls, pricing)
        key_counts = [t["usage"]["distinct_keys"] for t in h_trials if t.get("usage") and t["usage"].get("distinct_keys") is not None]
        growths = [t["usage"]["context_growth_per_turn"] for t in h_trials if t.get("usage") and t["usage"].get("context_growth_per_turn") is not None]
        warm_in = sum(
            sum(int(c.get("input_tokens") or 0) for c in [x for x in t["proxy_calls"] if not x.get("is_side_call")][1:])
            for t in h_trials
            if len([x for x in t["proxy_calls"] if not x.get("is_side_call")]) > 1
        )
        warm_cached = sum(
            sum(int(c.get("cached_tokens") or 0) for c in [x for x in t["proxy_calls"] if not x.get("is_side_call")][1:])
            for t in h_trials
            if len([x for x in t["proxy_calls"] if not x.get("is_side_call")]) > 1
        )
        total_breaks = sum(t["usage"]["cache_breaks"] for t in h_trials if t.get("usage"))
        usage_agg["warmup_cache_hit_rate"] = (warm_cached / warm_in) if warm_in else None
        usage_agg["cache_breaks"] = total_breaks
        usage_agg["avg_distinct_keys"] = (sum(key_counts) / len(key_counts)) if key_counts else None
        usage_agg["avg_context_growth_per_turn"] = (sum(growths) / len(growths)) if growths else None
    else:
        fallback_trials = [t["usage"] for t in h_trials if t.get("usage")]
        if fallback_trials:
            t_in = sum(u["total_input_tokens"] for u in fallback_trials)
            t_uncached = sum(u["uncached_input_tokens"] for u in fallback_trials)
            t_cached = sum(u["cache_read_tokens"] for u in fallback_trials)
            t_out = sum(u["output_tokens"] for u in fallback_trials)
            t_reason = sum(u["reasoning_tokens"] for u in fallback_trials)
            t_cost = sum(u["cost_usd"] for u in fallback_trials if u["cost_usd"] is not None)
            warm_in = sum(int(u.get("warmup_input_tokens") or 0) for u in fallback_trials)
            warm_cached = sum(int(u.get("warmup_cached_tokens") or 0) for u in fallback_trials)
            denom = t_uncached + t_cached
            usage_agg = {
                "calls": sum(u["calls"] for u in fallback_trials),
                "main_calls": sum(u["main_calls"] for u in fallback_trials),
                "cost_usd": t_cost,
                "uncached_input_tokens": t_uncached,
                "cache_read_tokens": t_cached,
                "cache_write_tokens": sum(u["cache_write_tokens"] for u in fallback_trials),
                "total_input_tokens": t_in,
                "output_tokens": t_out,
                "reasoning_tokens": t_reason,
                "cache_hit_rate": (t_cached / denom) if denom else None,
                "warmup_cache_hit_rate": (warm_cached / warm_in) if warm_in else None,
                "key_coverage": None,
                "cache_breaks": sum(u.get("cache_breaks", 0) for u in fallback_trials),
                "peak_context_tokens": max((u["peak_context_tokens"] for u in fallback_trials if u.get("peak_context_tokens") is not None), default=None),
                "avg_ttft_ms": None,
                "p95_ttft_ms": None,
                "llm_api_sec": sum(u.get("llm_api_sec", 0.0) for u in fallback_trials),
                "http_errors": 0,
                "rate_limits": 0,
                "efforts": {},
                "avg_distinct_keys": None,
                "avg_context_growth_per_turn": None,
            }
        else:
            usage_agg = _compute_cache_and_usage_stats([], pricing)
            usage_agg["avg_distinct_keys"] = None
            usage_agg["avg_context_growth_per_turn"] = None

    # Task pass@N & stability pass^N
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for t in scored:
        by_task[t["task_name"]].append(t)
    tasks_covered = len(by_task)
    tasks_any_pass = sum(1 for items in by_task.values() if any(x["outcome"] == "pass" for x in items))
    tasks_all_pass = sum(
        1 for items in by_task.values() if len(items) >= n_attempts and all(x["outcome"] == "pass" for x in items)
    )

    # Pass vs Fail token ratio
    pass_tokens = [
        (t["usage"]["uncached_input_tokens"] + t["usage"]["output_tokens"])
        for t in passes
        if t.get("usage")
    ]
    fail_tokens = [
        (t["usage"]["uncached_input_tokens"] + t["usage"]["output_tokens"])
        for t in (fails + timeouts)
        if t.get("usage")
    ]

    # Behavioral metrics
    turns_list = [t["behavior"]["turns"] for t in scored if t["behavior"].get("turns") is not None]
    tc_list = [t["behavior"]["tool_calls"] for t in scored if t["behavior"].get("tool_calls") is not None]
    twt_list = [t["behavior"]["turns_with_tools"] for t in scored if t["behavior"].get("turns_with_tools")]
    err_trials = [t["behavior"] for t in scored if t["behavior"].get("tool_errors") is not None and (t["behavior"].get("tool_calls") or 0) > 0]
    sc_trials = [t["behavior"]["self_checked"] for t in scored if t["behavior"].get("self_checked") is not None]
    dl_trials = [t["behavior"]["doom_loop"] for t in scored if t["behavior"].get("doom_loop") is not None]
    llm_secs = [t["behavior"]["llm_sec"] for t in scored if t["behavior"].get("llm_sec") is not None and t.get("agent_sec")]
    paired_agent_secs = [t["agent_sec"] for t in scored if t["behavior"].get("llm_sec") is not None and t.get("agent_sec")]
    stop_reasons = dict(Counter(t["behavior"]["stop_reason"] for t in scored if t["behavior"].get("stop_reason")))

    total_cost = usage_agg["cost_usd"] if (usage_agg["cost_usd"] is not None and (usage_agg["calls"] > 0 or usage_agg["total_input_tokens"] > 0)) else None
    pass_rate = (len(passes) / len(scored)) if scored else None

    # Slices: tb4 vs tb21 & category
    slices: dict[str, dict[str, Any]] = {}
    for slice_name, group in [
        ("tb4", [t for t in scored if manifest_map.get(t["task_name"], {}).get("subset") == "tb4"]),
        ("tb21", [t for t in scored if manifest_map.get(t["task_name"], {}).get("subset") == "tb21"]),
    ]:
        if group:
            gp = sum(1 for x in group if x["outcome"] == "pass")
            gc = [x["usage"]["cost_usd"] for x in group if x.get("usage") and x["usage"]["cost_usd"] is not None]
            gs = [x["run_sec"] for x in group if x["run_sec"] is not None]
            slices[slice_name] = {
                "scored": len(group),
                "passes": gp,
                "pass_rate": gp / len(group),
                "cost_per_pass_usd": (sum(gc) / gp) if (gc and gp) else None,
                "mean_sec": (sum(gs) / len(gs)) if gs else None,
            }

    total_out = usage_agg["output_tokens"]
    return {
        "harness": harness,
        "model": model,
        "scored": len(scored),
        "passes": len(passes),
        "fails": len(fails),
        "timeouts": len(timeouts),
        "infra": len(infra),
        "running": len(running),
        "pass_rate": pass_rate,
        "task_pass_at_n": (tasks_any_pass / tasks_covered) if tasks_covered else None,
        "task_pass_all_n": (tasks_all_pass / tasks_covered) if (tasks_covered and n_attempts >= 2) else None,
        "tasks_covered": tasks_covered,
        "tasks_any_pass": tasks_any_pass,
        "mean_sec": (sum(run_secs) / len(run_secs)) if run_secs else None,
        "mean_exec_sec": (sum(exec_secs) / len(exec_secs)) if exec_secs else None,
        "p50_exec_sec": percentile(exec_secs, 0.50),
        "p90_exec_sec": percentile(exec_secs, 0.90),
        "time_per_pass_sec": (sum(run_secs) / len(passes)) if (run_secs and passes) else None,
        "total_cost_usd": total_cost,
        "cost_per_attempt_usd": (total_cost / len(scored)) if (total_cost is not None and scored) else None,
        "cost_per_pass_usd": (total_cost / len(passes)) if (total_cost is not None and passes) else None,
        "usage": usage_agg,
        "reasoning_share": (usage_agg["reasoning_tokens"] / total_out) if total_out else None,
        "pass_avg_active_tokens": (sum(pass_tokens) / len(pass_tokens)) if pass_tokens else None,
        "fail_avg_active_tokens": (sum(fail_tokens) / len(fail_tokens)) if fail_tokens else None,
        "avg_turns": (sum(turns_list) / len(turns_list)) if turns_list else None,
        "avg_tool_calls": (sum(tc_list) / len(tc_list)) if tc_list else None,
        "parallel_tool_rate": (sum(tc_list) / sum(twt_list)) if (tc_list and sum(twt_list) > 0) else None,
        "tool_error_rate": (
            sum(b["tool_errors"] for b in err_trials) / sum(b["tool_calls"] for b in err_trials)
            if err_trials
            else None
        ),
        "self_check_rate": (sum(1 for x in sc_trials if x) / len(sc_trials)) if sc_trials else None,
        "doom_loop_rate": (sum(1 for x in dl_trials if x) / len(dl_trials)) if dl_trials else None,
        "llm_time_share": (sum(llm_secs) / sum(paired_agent_secs)) if (llm_secs and sum(paired_agent_secs) > 0) else None,
        "stop_reasons": stop_reasons,
        "slices": slices,
    }


def summarize_jobs(
    job_dirs: list[Path],
    pricing: dict[str, float],
    pricing_source: str,
    manifest_tasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute FrontierHarness v1.1 metrics, cache diagnostics, and multi-harness comparisons."""
    all_trials: list[dict[str, Any]] = []
    unassigned_by_job_harness: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    job_cfgs: list[dict[str, Any]] = []
    all_proxy_calls: list[dict[str, Any]] = []
    max_attempts = 1

    for jdir in job_dirs:
        meta = load_job_meta(jdir)
        job_cfg, trials = collect_job_trials(jdir)
        job_cfgs.append(job_cfg)
        max_attempts = max(max_attempts, int(job_cfg.get("n_attempts") or 1))
        proxy_rows = load_proxy_usage(jdir, meta)
        has_proxy_log = bool(proxy_rows)
        unassigned = attribute_proxy_calls(trials, proxy_rows)
        for tr in trials:
            _enrich_trial_telemetry(tr, pricing=pricing, has_proxy_log=has_proxy_log)
        for rec in unassigned:
            if rec.get("harness"):
                unassigned_by_job_harness[(jdir.name, str(rec["harness"]))].append(rec)
        all_trials.extend(trials)
        all_proxy_calls.extend(proxy_rows)

    if len(job_dirs) > 1:
        jobs_per_harness: dict[str, set[str]] = defaultdict(set)
        for tr in all_trials:
            jobs_per_harness[tr["raw_harness"]].add(tr["job_name"])
        colliding = {h for h, jnames in jobs_per_harness.items() if len(jnames) > 1}
        for tr in all_trials:
            if tr["raw_harness"] in colliding:
                tr["harness"] = f"{tr['raw_harness']}@{tr['job_name']}"

    planned_tasks, manifest_map = _resolve_planned_tasks(job_cfgs, all_trials, manifest_tasks)
    planned_trials = sum(
        len(planned_tasks) * int(cfg.get("n_attempts") or 1) * max(1, len(cfg.get("agents") or []))
        for cfg in job_cfgs
    )

    by_agent: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for tr in all_trials:
        by_agent[(tr["harness"], tr["raw_harness"], tr["model"])].append(tr)

    standings: list[dict[str, Any]] = []
    for (harness, raw_harness, model), h_trials in sorted(by_agent.items()):
        trial_jobs = {t["job_name"] for t in h_trials}
        extra = [
            rec
            for jn in trial_jobs
            for rec in unassigned_by_job_harness.get((jn, raw_harness), [])
        ]
        standings.append(
            _aggregate_harness_stats(
                harness=harness,
                model=model,
                h_trials=h_trials,
                unassigned_calls=extra,
                pricing=pricing,
                n_attempts=max_attempts,
                manifest_map=manifest_map,
            )
        )

    standings.sort(
        key=lambda x: (
            x["cost_per_pass_usd"] is None,
            x["cost_per_pass_usd"] if x["cost_per_pass_usd"] is not None else 0.0,
            -(x["pass_rate"] or 0.0),
        )
    )

    finished = [t for t in all_trials if t["outcome"] in {"pass", "fail", "timeout", "infra"}]
    scored = [t for t in finished if t["outcome"] != "infra"]
    passes = [t for t in scored if t["outcome"] == "pass"]
    fails = [t for t in scored if t["outcome"] == "fail"]
    timeouts = [t for t in scored if t["outcome"] == "timeout"]
    infra = [t for t in finished if t["outcome"] == "infra"]
    running = [t for t in all_trials if t["outcome"] == "running"]

    # Overall usage across standings (avoids dropping unassigned calls at concurrency > 1)
    total_cost_vals = [st["total_cost_usd"] for st in standings if st["total_cost_usd"] is not None]
    total_cost = sum(total_cost_vals) if total_cost_vals else None
    total_in = sum(st["usage"]["total_input_tokens"] for st in standings)
    total_uncached = sum(st["usage"]["uncached_input_tokens"] for st in standings)
    total_cached = sum(st["usage"]["cache_read_tokens"] for st in standings)
    total_out = sum(st["usage"]["output_tokens"] for st in standings)
    total_reason = sum(st["usage"]["reasoning_tokens"] for st in standings)
    total_calls = sum(st["usage"]["calls"] for st in standings)
    denom = total_uncached + total_cached

    overall_efforts: Counter[str] = Counter()
    for st in standings:
        overall_efforts.update(st["usage"].get("efforts") or {})

    run_secs = [t["run_sec"] for t in scored if t["run_sec"] is not None]
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for tr in all_trials:
        by_task[tr["task_name"]].append(tr)

    harness_names = [st["harness"] for st in standings]
    primary_harness = next((h for h in harness_names if "mizar" in h.lower()), harness_names[0] if harness_names else None)

    quadrant = {"all_pass": 0, "primary_only_pass": 0, "primary_only_fail": 0, "all_fail": 0}
    task_rows = []
    tasks_any_pass = 0
    tasks_with_scored = 0
    for task_name in planned_tasks:
        items = by_task.get(task_name, [])
        t_scored = [x for x in items if x["outcome"] in {"pass", "fail", "timeout"}]
        t_pass = [x for x in t_scored if x["outcome"] == "pass"]
        if t_scored:
            tasks_with_scored += 1
        if t_pass:
            tasks_any_pass += 1
        t_secs = [x["run_sec"] for x in t_scored if x["run_sec"] is not None]
        t_costs = [x["usage"]["cost_usd"] for x in items if x.get("usage") and x["usage"].get("cost_usd") is not None]

        per_harness: dict[str, dict[str, Any]] = {}
        for h in harness_names:
            hi = [x for x in items if x["harness"] == h]
            if not hi:
                continue
            h_sc = [x for x in hi if x["outcome"] in {"pass", "fail", "timeout"}]
            h_p = sum(1 for x in h_sc if x["outcome"] == "pass")
            h_secs = [
                (x["agent_sec"] if x["agent_sec"] is not None else x["run_sec"])
                for x in h_sc
                if (x["agent_sec"] is not None or x["run_sec"] is not None)
            ]
            h_cost = [x["usage"]["cost_usd"] for x in hi if x.get("usage") and x["usage"].get("cost_usd") is not None]
            h_in = sum(x["usage"]["total_input_tokens"] for x in hi if x.get("usage"))
            h_cached = sum(x["usage"]["cache_read_tokens"] for x in hi if x.get("usage"))
            per_harness[h] = {
                "attempts": [x["outcome"] for x in hi],
                "passes": h_p,
                "scored": len(h_sc),
                "mean_sec": (sum(h_secs) / len(h_secs)) if h_secs else None,
                "cost_usd": sum(h_cost) if h_cost else None,
                "cache_hit_rate": (h_cached / h_in) if h_in else None,
                "turns": next((x["behavior"]["turns"] for x in hi if x["behavior"].get("turns") is not None), None),
            }

        if len(harness_names) >= 2 and primary_harness and all(h in per_harness and per_harness[h]["scored"] > 0 for h in harness_names):
            p_ok = per_harness[primary_harness]["passes"] > 0
            others_ok = [per_harness[h]["passes"] > 0 for h in harness_names if h != primary_harness]
            if p_ok and all(others_ok):
                quadrant["all_pass"] += 1
            elif p_ok and not any(others_ok):
                quadrant["primary_only_pass"] += 1
            elif not p_ok and any(others_ok):
                quadrant["primary_only_fail"] += 1
            elif not p_ok and not any(others_ok):
                quadrant["all_fail"] += 1

        meta_t = manifest_map.get(task_name, {})
        task_rows.append(
            {
                "task": task_name,
                "subset": meta_t.get("subset", "-"),
                "category": meta_t.get("category", "Other"),
                "attempts": [x["outcome"] for x in items],
                "passes": len(t_pass),
                "scored": len(t_scored),
                "infra": sum(1 for x in items if x["outcome"] == "infra"),
                "running": sum(1 for x in items if x["outcome"] == "running"),
                "pass_rate": (len(t_pass) / len(t_scored)) if t_scored else None,
                "mean_sec": (sum(t_secs) / len(t_secs)) if t_secs else None,
                "cost_usd": sum(t_costs) if t_costs else None,
                "errors": [f"{x['error_type']}: {x['error_message']}" for x in items if x.get("error_type")],
                "per_harness": per_harness,
            }
        )

    # Round-by-round breakdown
    by_task_agent: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for tr in all_trials:
        by_task_agent[(tr["task_name"], tr["harness"], tr["model"])].append(tr)
    rounds: dict[int, dict[str, Any]] = {}
    for idx in range(max_attempts):
        r_items = [items[idx] for items in by_task_agent.values() if len(items) > idx]
        r_scored = [x for x in r_items if x["outcome"] in {"pass", "fail", "timeout"}]
        r_pass = [x for x in r_scored if x["outcome"] == "pass"]
        r_secs = [x["run_sec"] for x in r_scored if x["run_sec"] is not None]
        rounds[idx + 1] = {
            "n_scored": len(r_scored),
            "passes": len(r_pass),
            "pass_rate": (len(r_pass) / len(r_scored)) if r_scored else None,
            "mean_sec": (sum(r_secs) / len(r_secs)) if r_secs else None,
        }

    # Category slice table across harnesses
    categories = sorted({r["category"] for r in task_rows if r["scored"] > 0})
    category_slices: list[dict[str, Any]] = []
    for cat in categories:
        cat_tasks = {r["task"] for r in task_rows if r["category"] == cat}
        row_entry: dict[str, Any] = {"category": cat, "tasks": len(cat_tasks), "by_harness": {}}
        for h in harness_names:
            h_sc = [t for t in scored if t["harness"] == h and t["task_name"] in cat_tasks]
            if h_sc:
                hp = sum(1 for x in h_sc if x["outcome"] == "pass")
                row_entry["by_harness"][h] = {"passes": hp, "scored": len(h_sc), "pass_rate": hp / len(h_sc)}
        category_slices.append(row_entry)

    job_label = job_dirs[0].name if len(job_dirs) == 1 else " vs ".join(d.name for d in job_dirs)
    return {
        "job_name": job_label,
        "source_jobs": [d.name for d in job_dirs],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pricing_per_1m_usd": pricing,
        "pricing_source": pricing_source,
        "planned_trials": planned_trials,
        "n_attempts": max_attempts,
        "primary_harness": primary_harness,
        "quadrant": quadrant,
        "counts": {
            "started": len(all_trials),
            "finished": len(finished),
            "scored": len(scored),
            "pass": len(passes),
            "fail": len(fails),
            "timeout": len(timeouts),
            "infra": len(infra),
            "running": len(running),
            "tasks_covered": tasks_with_scored,
            "tasks_any_pass": tasks_any_pass,
            "tasks_planned": len(planned_tasks),
        },
        "metrics": {
            "pass_rate": (len(passes) / len(scored)) if scored else None,
            "task_pass_at_n": (tasks_any_pass / tasks_with_scored) if tasks_with_scored else None,
            "mean_time_sec": (sum(run_secs) / len(run_secs)) if run_secs else None,
            "time_per_pass_sec": (sum(run_secs) / len(passes)) if (run_secs and passes) else None,
            "total_cost_usd": total_cost,
            "cost_per_attempt_usd": (total_cost / len(scored)) if (total_cost is not None and scored) else None,
            "cost_per_pass_usd": (total_cost / len(passes)) if (total_cost is not None and passes) else None,
        },
        "usage": {
            "calls": total_calls,
            "attributed_trial_calls": sum(len(t["proxy_calls"]) for t in all_trials),
            "proxy_calls_total": len(all_proxy_calls),
            "cost_usd": total_cost,
            "uncached_input_tokens": total_uncached,
            "cache_read_tokens": total_cached,
            "total_input_tokens": total_in,
            "output_tokens": total_out,
            "reasoning_tokens": total_reason,
            "cache_hit_rate": (total_cached / denom) if denom else None,
            "efforts": dict(overall_efforts),
        },
        "standings": standings,
        "infra_breakdown": dict(Counter(t["error_type"] for t in infra if t["error_type"])),
        "rounds": rounds,
        "category_slices": category_slices,
        "tasks": task_rows,
    }


def fmt_pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.1f}%"


def fmt_num(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "-"
    if digits == 0:
        return f"{int(round(value)):,}"
    return f"{value:,.{digits}f}"


def fmt_ms(value: float | None) -> str:
    return "-" if value is None else f"{int(round(value)):,}ms"


def fmt_tok(value: int | float | None) -> str:
    if value is None:
        return "-"
    v = float(value)
    if v >= 1_000_000:
        return f"{v / 1_000_000:.2f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{int(round(v))}"


def fmt_dur(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    s = int(round(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def fmt_usd(value: float | None) -> str:
    return "-" if value is None else f"${value:.4f}" if value < 1 else f"${value:.2f}"


def render_markdown(summary: dict[str, Any]) -> str:
    c = summary["counts"]
    m = summary["metrics"]
    u = summary["usage"]
    p = summary["pricing_per_1m_usd"]
    badge = {"pass": "P", "fail": "F", "timeout": "T", "infra": "I", "running": "R"}
    harnesses = [st["harness"] for st in summary["standings"]]

    lines = [
        f"# FrontierHarness Eval Score Report · `{summary['job_name']}`",
        "",
        f"- Generated: `{summary['generated_at']}`",
        f"- Trials: `{c['scored']}` scored / `{summary['planned_trials']}` planned (`{c['infra']}` infra errors, `{c['running']}` running)",
        f"- Pricing (`{summary['pricing_source']}`): `${p['uncached_input']}/1M` uncached in, `${p['cached_input']}/1M` cached in, `${p['output']}/1M` out",
        "",
        "## FrontierHarness Standings (Cost per pass ascending)",
        "",
        "| # | Harness | Model | Cost/pass | Pass rate | Pass@N | Cache hit | Warmup hit | Time/pass | Mean exec | Uncached in | Out | Timeout | Infra |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, st in enumerate(summary["standings"], 1):
        su = st["usage"]
        lines.append(
            f"| {rank} | `{st['harness']}` | `{st['model']}` | {fmt_usd(st['cost_per_pass_usd'])} | "
            f"{fmt_pct(st['pass_rate'])} ({st['passes']}/{st['scored']}) | {fmt_pct(st['task_pass_at_n'])} | "
            f"{fmt_pct(su['cache_hit_rate'])} | {fmt_pct(su['warmup_cache_hit_rate'])} | "
            f"{fmt_dur(st['time_per_pass_sec'])} | {fmt_dur(st['mean_exec_sec'])} | "
            f"{fmt_tok(su['uncached_input_tokens'])} | {fmt_tok(su['output_tokens'])} | "
            f"{st['timeouts']} | {st['infra']} |"
        )

    lines.extend(
        [
            "",
            "## Harness Diagnostics & Efficiency",
            "",
            "| Harness | Key cov | Cache breaks | Peak ctx | Growth/turn | TTFT P50 | LLM time % | Reason % | Turns/trial | Tools/trial | Parallel | Tool err % | Self-check | Doom-loop | Pass tok | Fail tok | HTTP err |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for st in summary["standings"]:
        su = st["usage"]
        lines.append(
            f"| `{st['harness']}` | {fmt_pct(su.get('key_coverage'))} | {su.get('cache_breaks', 0)} | "
            f"{fmt_tok(su.get('peak_context_tokens'))} | {fmt_tok(su.get('avg_context_growth_per_turn'))} | "
            f"{fmt_ms(su.get('avg_ttft_ms'))} | {fmt_pct(st.get('llm_time_share'))} | {fmt_pct(st.get('reasoning_share'))} | "
            f"{fmt_num(st.get('avg_turns'), 1)} | {fmt_num(st.get('avg_tool_calls'), 1)} | {fmt_num(st.get('parallel_tool_rate'), 2)} | "
            f"{fmt_pct(st.get('tool_error_rate'))} | {fmt_pct(st.get('self_check_rate'))} | {fmt_pct(st.get('doom_loop_rate'))} | "
            f"{fmt_tok(st.get('pass_avg_active_tokens'))} | {fmt_tok(st.get('fail_avg_active_tokens'))} | {su.get('http_errors', 0)} |"
        )

    stop_lines = [
        f"- `{st['harness']}`: {st['stop_reasons']}"
        for st in summary["standings"]
        if st.get("stop_reasons")
    ]
    if stop_lines:
        lines.extend(["", "### Stop Reasons", "", *stop_lines])

    if summary.get("category_slices"):
        cat_hdr = " | ".join(f"`{h}`" for h in harnesses)
        cat_sep = " | ".join("---:" for _ in harnesses)
        lines.extend(
            [
                "",
                "## Category Slices",
                "",
                f"| Category | Tasks | {cat_hdr} |",
                f"| --- | ---: | {cat_sep} |",
            ]
        )
        for cs in summary["category_slices"]:
            cat_cells = []
            for h in harnesses:
                ch = cs["by_harness"].get(h)
                cat_cells.append(f"{fmt_pct(ch['pass_rate'])} ({ch['passes']}/{ch['scored']})" if ch else "-")
            lines.append(f"| {cs['category']} | {cs['tasks']} | {' | '.join(cat_cells)} |")

    if any(st["slices"] for st in summary["standings"]):
        lines.extend(
            [
                "",
                "## Dataset Slices (`tb4` vs `tb21`)",
                "",
                "| Harness | `tb4` Pass rate | `tb4` Cost/pass | `tb21` Pass rate | `tb21` Cost/pass |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for st in summary["standings"]:
            tb4 = st["slices"].get("tb4") or {}
            tb21 = st["slices"].get("tb21") or {}
            lines.append(
                f"| `{st['harness']}` | {fmt_pct(tb4.get('pass_rate'))} | {fmt_usd(tb4.get('cost_per_pass_usd'))} | "
                f"{fmt_pct(tb21.get('pass_rate'))} | {fmt_usd(tb21.get('cost_per_pass_usd'))} |"
            )

    if len(harnesses) >= 2 and summary.get("primary_harness"):
        q = summary["quadrant"]
        ph = summary["primary_harness"]
        lines.extend(
            [
                "",
                f"## Head-to-Head Quadrant (Primary: `{ph}`)",
                "",
                f"- **All Pass**: `{q['all_pass']}` | **`{ph}` Only Pass**: `{q['primary_only_pass']}` | **`{ph}` Only Fail**: `{q['primary_only_fail']}` | **All Fail**: `{q['all_fail']}`",
            ]
        )

    lines.extend(
        [
            "",
            "## Overall Aggregate",
            "",
            "| Metric | Value | Notes |",
            "| --- | --- | --- |",
            f"| Pass rate | **{fmt_pct(m['pass_rate'])}** | `{c['pass']} / {c['scored']}` scored attempts |",
            f"| Task pass@{summary['n_attempts']} | **{fmt_pct(m['task_pass_at_n'])}** | `{c['tasks_any_pass']} / {c['tasks_covered']}` tasks covered (of {c['tasks_planned']} planned) |",
            f"| Total cost | `{fmt_usd(m['total_cost_usd'])}` | `{u['calls']}` logged API calls ({u.get('efforts') or '-'}) |",
            f"| Cache hit rate | `{fmt_pct(u['cache_hit_rate'])}` | `uncached={u['uncached_input_tokens']:,}` / `cached={u['cache_read_tokens']:,}` / `out={u['output_tokens']:,}` (reasoning `{u['reasoning_tokens']:,}`) |",
        ]
    )

    if len(summary["rounds"]) > 1:
        lines.extend(["", "## Rounds", "", "| Round | Scored | Passes | Pass rate | Mean time |", "| --- | ---: | ---: | ---: | ---: |"])
        for r_idx, r in sorted(summary["rounds"].items()):
            lines.append(f"| Attempt {r_idx} | {r['n_scored']} | {r['passes']} | {fmt_pct(r['pass_rate'])} | {fmt_dur(r['mean_sec'])} |")

    if summary["infra_breakdown"]:
        lines.extend(["", "## Infra Errors", ""])
        for etype, count in sorted(summary["infra_breakdown"].items(), key=lambda kv: -kv[1]):
            lines.append(f"- `{etype}`: `{count}`")

    lines.extend(["", "## Task Breakdown", ""])
    if len(harnesses) >= 2:
        hdr_cols = " | ".join(f"`{h}`" for h in harnesses)
        sep_cols = " | ".join("---:" for _ in harnesses)
        lines.append(f"| Task | Subset | Category | {hdr_cols} | Notes |")
        lines.append(f"| --- | --- | --- | {sep_cols} | --- |")
        for row in summary["tasks"]:
            cells = []
            for h in harnesses:
                ph = row["per_harness"].get(h)
                if not ph:
                    cells.append("-")
                else:
                    seq = "".join(badge.get(o, "?") for o in ph["attempts"])
                    cells.append(f"{seq} ({fmt_dur(ph['mean_sec'])}, {fmt_usd(ph['cost_usd'])}, {fmt_pct(ph['cache_hit_rate'])})")
            errs = "; ".join(row["errors"][:2]) if row["errors"] else "-"
            lines.append(f"| `{row['task']}` | {row['subset']} | {row['category']} | {' | '.join(cells)} | {errs} |")
    else:
        lines.append("| Task | Subset | Attempts | Score | Mean time | Cost | Errors |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | --- |")
        for row in summary["tasks"]:
            seq = " ".join(badge.get(o, "?") for o in row["attempts"]) or "-"
            errs = "; ".join(row["errors"][:2]) if row["errors"] else "-"
            lines.append(
                f"| `{row['task']}` | {row['subset']} | `{seq}` | {row['passes']}/{row['scored']} | "
                f"{fmt_dur(row['mean_sec'])} | {fmt_usd(row['cost_usd'])} | {errs} |"
            )
    lines.append("")
    return "\n".join(lines)


def render_html(summary: dict[str, Any]) -> str:
    c = summary["counts"]
    m = summary["metrics"]
    u = summary["usage"]
    harnesses = [st["harness"] for st in summary["standings"]]
    primary = summary.get("primary_harness")
    multi = len(harnesses) >= 2

    pill_map = {
        "pass": ("P", "pill-pass"),
        "fail": ("F", "pill-fail"),
        "timeout": ("T", "pill-timeout"),
        "infra": ("I", "pill-infra"),
        "running": ("R", "pill-run"),
    }

    standings_html = []
    diag_html = []
    for rank, st in enumerate(summary["standings"], 1):
        su = st["usage"]
        standings_html.append(
            f"<tr>"
            f'<td class="num mono">{rank}</td>'
            f'<td class="mono">{html.escape(st["harness"])}</td>'
            f'<td class="mono">{html.escape(st["model"])}</td>'
            f'<td class="num mono">{fmt_usd(st["cost_per_pass_usd"])}</td>'
            f'<td class="num mono">{fmt_pct(st["pass_rate"])}</td>'
            f'<td class="num mono">{fmt_pct(st["task_pass_at_n"])}</td>'
            f'<td class="num mono">{fmt_pct(su["cache_hit_rate"])}</td>'
            f'<td class="num mono">{fmt_pct(su["warmup_cache_hit_rate"])}</td>'
            f'<td class="num mono">{fmt_dur(st["time_per_pass_sec"])}</td>'
            f'<td class="num mono">{fmt_dur(st["mean_exec_sec"])}</td>'
            f'<td class="num mono">{fmt_tok(su["uncached_input_tokens"])}</td>'
            f'<td class="num mono">{fmt_tok(su["output_tokens"])}</td>'
            f'<td class="num mono">{st["passes"]}/{st["scored"]}</td>'
            f'<td class="num mono">{st["timeouts"]}/{st["infra"]}</td>'
            f"</tr>"
        )
        diag_html.append(
            f"<tr>"
            f'<td class="mono">{html.escape(st["harness"])}</td>'
            f'<td class="num mono">{fmt_pct(su.get("key_coverage"))}</td>'
            f'<td class="num mono">{su.get("cache_breaks", 0)}</td>'
            f'<td class="num mono">{fmt_tok(su.get("peak_context_tokens"))}</td>'
            f'<td class="num mono">{fmt_tok(su.get("avg_context_growth_per_turn"))}</td>'
            f'<td class="num mono">{fmt_ms(su.get("avg_ttft_ms"))}</td>'
            f'<td class="num mono">{fmt_pct(st.get("llm_time_share"))}</td>'
            f'<td class="num mono">{fmt_pct(st.get("reasoning_share"))}</td>'
            f'<td class="num mono">{fmt_num(st.get("avg_turns"), 1)}</td>'
            f'<td class="num mono">{fmt_num(st.get("avg_tool_calls"), 1)}</td>'
            f'<td class="num mono">{fmt_num(st.get("parallel_tool_rate"), 2)}</td>'
            f'<td class="num mono">{fmt_pct(st.get("tool_error_rate"))}</td>'
            f'<td class="num mono">{su.get("http_errors", 0)}</td>'
            f"</tr>"
        )

    rows_html = []
    for row in summary["tasks"]:
        status_cls = (
            "pass"
            if row["passes"] > 0
            else "infra"
            if row["infra"] > 0 and row["scored"] == 0
            else "pending"
            if row["scored"] == 0
            else "fail"
        )
        primary_loss = "0"
        if multi and primary and primary in row["per_harness"]:
            p_pass = row["per_harness"][primary]["passes"] > 0
            o_pass = any(row["per_harness"].get(h, {}).get("passes", 0) > 0 for h in harnesses if h != primary)
            if not p_pass and o_pass:
                primary_loss = "1"
        err_text = html.escape("; ".join(row["errors"][:2])) if row["errors"] else ""

        if multi:
            h_cells = []
            for h in harnesses:
                ph = row["per_harness"].get(h)
                if not ph:
                    h_cells.append('<td class="muted">-</td>')
                else:
                    pills = "".join(
                        f'<span class="pill {pill_map.get(o, ("?", ""))[1]}">{pill_map.get(o, ("?", ""))[0]}</span>'
                        for o in ph["attempts"]
                    )
                    sub = f"{fmt_dur(ph['mean_sec'])} · {fmt_usd(ph['cost_usd'])} · {fmt_pct(ph['cache_hit_rate'])}"
                    h_cells.append(f"<td>{pills}<div class=\"subcell mono\">{sub}</div></td>")
            rows_html.append(
                f'<tr data-status="{status_cls}" data-ploss="{primary_loss}">'
                f'<td class="mono">{html.escape(row["task"])}</td>'
                f'<td class="mono">{html.escape(row["subset"])}</td>'
                f'<td>{html.escape(row["category"])}</td>'
                f'{"".join(h_cells)}'
                f'<td class="err">{err_text}</td>'
                f"</tr>"
            )
        else:
            pills = "".join(
                f'<span class="pill {pill_map.get(o, ("?", ""))[1]}">{pill_map.get(o, ("?", ""))[0]}</span>'
                for o in row["attempts"]
            ) or '<span class="muted">pending</span>'
            rows_html.append(
                f'<tr data-status="{status_cls}" data-ploss="0">'
                f'<td class="mono">{html.escape(row["task"])}</td>'
                f'<td class="mono">{html.escape(row["subset"])}</td>'
                f"<td>{pills}</td>"
                f'<td class="num mono">{row["passes"]}/{row["scored"]}</td>'
                f'<td class="num mono">{fmt_pct(row["pass_rate"])}</td>'
                f'<td class="num mono">{fmt_dur(row["mean_sec"])}</td>'
                f'<td class="num mono">{fmt_usd(row["cost_usd"])}</td>'
                f'<td class="err">{err_text}</td>'
                f"</tr>"
            )

    task_headers = (
        f"<th>Task</th><th>Subset</th><th>Category</th>{''.join(f'<th>{html.escape(h)}</th>' for h in harnesses)}<th>Errors</th>"
        if multi
        else '<th>Task</th><th>Subset</th><th>Attempts</th><th class="num">Score</th><th class="num">Pass rate</th><th class="num">Mean time</th><th class="num">Cost</th><th>Errors</th>'
    )
    ploss_btn = '<button onclick="filterRows(\'ploss\',this)">Primary Only Fail</button>' if multi else ""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FrontierHarness Eval · {html.escape(summary["job_name"])}</title>
<style>
  :root {{ --bg:#0c0d10; --card:#151820; --ink:#eef1f6; --muted:#8e97a8; --line:#242936; --pass:#22c55e; --fail:#ef4444; --timeout:#f59e0b; --infra:#a855f7; --run:#3b82f6; }}
  body {{ margin:0; padding:24px; background:var(--bg); color:var(--ink); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1360px; margin:0 auto; }}
  h1 {{ margin:0 0 6px; font-size:20px; }}
  h2 {{ margin:20px 0 10px; font-size:15px; color:var(--muted); font-weight:600; }}
  .sub {{ color:var(--muted); margin-bottom:20px; font-size:13px; }}
  .kpis {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; margin-bottom:20px; }}
  .kpi {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
  .kpi .k {{ color:var(--muted); font-size:12px; }}
  .kpi .v {{ font-size:22px; font-weight:600; margin-top:2px; font-variant-numeric:tabular-nums; }}
  .kpi .s {{ color:var(--muted); font-size:12px; margin-top:2px; }}
  .filters {{ display:flex; gap:8px; margin-bottom:12px; flex-wrap:wrap; }}
  .filters button {{ background:var(--card); color:var(--ink); border:1px solid var(--line); border-radius:6px; padding:6px 12px; cursor:pointer; }}
  .filters button.active {{ border-color:#60a5fa; color:#60a5fa; }}
  .table-wrap {{ background:var(--card); border:1px solid var(--line); border-radius:10px; overflow:auto; margin-bottom:18px; }}
  table {{ width:100%; border-collapse:collapse; }}
  th,td {{ padding:8px 10px; border-bottom:1px solid var(--line); text-align:left; font-size:13px; vertical-align:top; }}
  th {{ color:var(--muted); font-weight:500; white-space:nowrap; }}
  .num {{ text-align:right; }}
  .mono {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-variant-numeric:tabular-nums; }}
  .subcell {{ font-size:11px; color:var(--muted); margin-top:2px; white-space:nowrap; }}
  .pill {{ display:inline-block; min-width:20px; text-align:center; border-radius:4px; padding:1px 5px; margin-right:4px; font-size:11px; font-weight:600; }}
  .pill-pass {{ background:rgba(34,197,94,.18); color:var(--pass); }}
  .pill-fail {{ background:rgba(239,68,68,.18); color:var(--fail); }}
  .pill-timeout {{ background:rgba(245,158,11,.18); color:var(--timeout); }}
  .pill-infra {{ background:rgba(168,85,247,.18); color:var(--infra); }}
  .pill-run {{ background:rgba(59,130,246,.18); color:var(--run); }}
  .err {{ color:var(--muted); max-width:300px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
  .muted {{ color:var(--muted); }}
</style>
</head>
<body>
<div class="wrap">
  <h1>FrontierHarness Eval · {html.escape(summary["job_name"])}</h1>
  <div class="sub">Generated {html.escape(summary["generated_at"])} · {c["scored"]} scored / {summary["planned_trials"]} planned · {c["infra"]} infra · {c["running"]} running</div>
  <div class="kpis">
    <div class="kpi"><div class="k">Pass rate (attempts)</div><div class="v">{fmt_pct(m["pass_rate"])}</div><div class="s">{c["pass"]} / {c["scored"]} scored</div></div>
    <div class="kpi"><div class="k">Task pass@{summary["n_attempts"]}</div><div class="v">{fmt_pct(m["task_pass_at_n"])}</div><div class="s">{c["tasks_any_pass"]} / {c["tasks_covered"]} tasks</div></div>
    <div class="kpi"><div class="k">Cost per pass</div><div class="v">{fmt_usd(m["cost_per_pass_usd"])}</div><div class="s">Total {fmt_usd(m["total_cost_usd"])}</div></div>
    <div class="kpi"><div class="k">Time per pass</div><div class="v">{fmt_dur(m["time_per_pass_sec"])}</div><div class="s">Mean {fmt_dur(m["mean_time_sec"])}/attempt</div></div>
    <div class="kpi"><div class="k">Cache hit rate</div><div class="v">{fmt_pct(u["cache_hit_rate"])}</div><div class="s">{u["calls"]} proxy calls</div></div>
    <div class="kpi"><div class="k">Timeouts / Infra</div><div class="v">{c["timeout"]} / {c["infra"]}</div><div class="s">Timeouts stay in denominator</div></div>
  </div>
  <h2>FrontierHarness Standings</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th class="num">#</th><th>Harness</th><th>Model</th><th class="num">Cost/pass</th><th class="num">Pass rate</th><th class="num">Pass@N</th><th class="num">Cache hit</th><th class="num">Warmup hit</th><th class="num">Time/pass</th><th class="num">Mean exec</th><th class="num">Uncached in</th><th class="num">Out</th><th class="num">Scored</th><th class="num">T/I</th></tr></thead>
      <tbody>{"".join(standings_html)}</tbody>
    </table>
  </div>
  <h2>Harness Diagnostics & Prompt Cache Health</h2>
  <div class="table-wrap">
    <table>
      <thead><tr><th>Harness</th><th class="num">Key cov</th><th class="num">Cache breaks</th><th class="num">Peak ctx</th><th class="num">Growth/turn</th><th class="num">TTFT avg</th><th class="num">LLM time</th><th class="num">Reason %</th><th class="num">Turns/trial</th><th class="num">Tools/trial</th><th class="num">Parallel</th><th class="num">Tool err</th><th class="num">HTTP err</th></tr></thead>
      <tbody>{"".join(diag_html)}</tbody>
    </table>
  </div>
  <div class="filters">
    <button class="active" onclick="filterRows('all',this)">All ({len(summary["tasks"])})</button>
    <button onclick="filterRows('pass',this)">Any Pass</button>
    <button onclick="filterRows('fail',this)">0 Pass</button>
    {ploss_btn}
    <button onclick="filterRows('infra',this)">Infra Only</button>
    <button onclick="filterRows('pending',this)">Pending</button>
  </div>
  <div class="table-wrap">
    <table id="tasks-table">
      <thead><tr>{task_headers}</tr></thead>
      <tbody>{"".join(rows_html)}</tbody>
    </table>
  </div>
</div>
<script>
function filterRows(mode, btn) {{
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.querySelectorAll('#tasks-table tbody tr').forEach(tr => {{
    if (mode === 'all') tr.style.display = '';
    else if (mode === 'ploss') tr.style.display = (tr.dataset.ploss === '1') ? '' : 'none';
    else tr.style.display = (tr.dataset.status === mode) ? '' : 'none';
  }});
}}
</script>
</body>
</html>
"""


def write_report_bundle(
    job_dirs: list[Path],
    out_dir: Path,
    pricing: dict[str, float],
    pricing_source: str,
) -> dict[str, Any]:
    manifest_tasks = load_manifest_tasks()
    summary = summarize_jobs(job_dirs, pricing=pricing, pricing_source=pricing_source, manifest_tasks=manifest_tasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (out_dir / "SCORE_REPORT.md").write_text(render_markdown(summary), encoding="utf-8")
    (out_dir / "live_report.html").write_text(render_html(summary), encoding="utf-8")
    return summary


def resolve_job_dir(target: str) -> Path:
    path = Path(target)
    if not path.is_absolute():
        candidate = JOBS_DIR / target
        path = candidate if candidate.exists() else (ROOT / target)
    path = path.resolve()
    if not (path / "config.json").is_file():
        raise FileNotFoundError(f"Job config.json not found in {path}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jobs", nargs="+", help="One or more job names under jobs/ or paths to job directories")
    parser.add_argument("--out-dir", help="Output directory (default: job dir for 1 job, jobs/compare-... for >1)")
    parser.add_argument("--price-in", type=float, help="Override USD per 1M uncached input tokens")
    parser.add_argument("--price-cached", type=float, help="Override USD per 1M cached input tokens")
    parser.add_argument("--price-out", type=float, help="Override USD per 1M output tokens")
    parser.add_argument("--price-cache-write", type=float, help="Override USD per 1M cache-write tokens")
    parser.add_argument("--watch", type=int, default=0, help="Refresh interval in seconds (0 = run once)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    job_dirs = [resolve_job_dir(j) for j in args.jobs]
    metas = [load_job_meta(d) for d in job_dirs]
    pricing, pricing_source = resolve_pricing(args, metas)

    if args.out_dir:
        out_dir = Path(args.out_dir).resolve()
    elif len(job_dirs) == 1:
        out_dir = job_dirs[0]
    else:
        joined = "-vs-".join(d.name for d in job_dirs)[:80]
        out_dir = JOBS_DIR / f"compare-{joined}"

    while True:
        summary = write_report_bundle(job_dirs, out_dir=out_dir, pricing=pricing, pricing_source=pricing_source)
        c, m, u = summary["counts"], summary["metrics"], summary["usage"]
        print(
            f"[{summary['job_name']}] scored={c['scored']}/{summary['planned_trials']} "
            f"pass={c['pass']} ({fmt_pct(m['pass_rate'])}) "
            f"task_pass@{summary['n_attempts']}={c['tasks_any_pass']}/{c['tasks_covered']} ({fmt_pct(m['task_pass_at_n'])}) "
            f"timeout={c['timeout']} infra={c['infra']} running={c['running']} "
            f"cost={fmt_usd(m['total_cost_usd'])} (per_pass={fmt_usd(m['cost_per_pass_usd'])}) "
            f"cache_hit={fmt_pct(u['cache_hit_rate'])}",
            flush=True,
        )
        print(f"  -> {out_dir / 'SCORE_REPORT.md'}", flush=True)
        print(f"  -> {out_dir / 'live_report.html'}", flush=True)
        if not args.watch:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
