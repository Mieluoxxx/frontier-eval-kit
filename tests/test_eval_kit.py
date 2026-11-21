#!/usr/bin/env python3
"""Unit tests for eval_proxy, run_eval, and score_report."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import eval_proxy
import run_eval
import score_report


class EvalProxyTests(unittest.TestCase):
    def test_extract_route_context(self) -> None:
        h, t, p = eval_proxy.extract_route_context("/a/mizar-v2/t/uuid-123/v1/responses?x=1")
        self.assertEqual((h, t, p), ("mizar-v2", "uuid-123", "/v1/responses?x=1"))

        h2, t2, p2 = eval_proxy.extract_route_context("/a/pi/v1/chat/completions")
        self.assertEqual((h2, t2, p2), ("pi", None, "/v1/chat/completions"))

        h3, t3, p3 = eval_proxy.extract_route_context("/v1/responses")
        self.assertEqual((h3, t3, p3), (None, None, "/v1/responses"))

    def test_inspect_and_rewrite_responses_medium_and_side_call(self) -> None:
        raw = json.dumps(
            {
                "model": "alias-model",
                "prompt_cache_key": "sess-abc",
                "input": [{"role": "user", "content": [{"type": "input_text", "text": "Fix bug X"}]}],
                "tools": [{"type": "function", "name": "bash"}],
                "max_output_tokens": 8000,
                "reasoning": {"effort": "high"},
            }
        ).encode("utf-8")
        out_bytes, meta = eval_proxy.inspect_and_rewrite_payload(
            "/v1/responses", raw, effort="medium", max_tokens=131072, force_model="isok/gpt-6.1-terra"
        )
        out = json.loads(out_bytes)
        self.assertEqual(out["model"], "isok/gpt-6.1-terra")
        self.assertEqual(out["reasoning"]["effort"], "medium")
        self.assertEqual(out["max_output_tokens"], 131072)
        self.assertFalse(meta["is_side_call"])
        self.assertEqual(meta["prompt_cache_key"], "sess-abc")
        self.assertEqual(meta["tool_count"], 1)
        self.assertIsNotNone(meta["user_prompt_sha256"])

        side_raw = json.dumps({"model": "x", "max_output_tokens": 512, "input": "hi"}).encode("utf-8")
        side_bytes, side_meta = eval_proxy.inspect_and_rewrite_payload(
            "/v1/responses", side_raw, effort="medium", max_tokens=131072
        )
        self.assertEqual(json.loads(side_bytes)["max_output_tokens"], 512)
        self.assertTrue(side_meta["is_side_call"])

    def test_sse_telemetry_sniffer_responses_usage_and_tools(self) -> None:
        sniffer = eval_proxy.SSETelemetrySniffer(start_mono=10.0)
        events = [
            {"type": "response.output_item.added", "item": {"id": "fc_1", "type": "function_call", "name": "bash"}},
            {"type": "response.output_text.delta", "delta": "hi"},
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_999",
                    "status": "completed",
                    "output": [
                        {"type": "function_call", "name": "bash"},
                        {"type": "function_call", "name": "read"},
                    ],
                    "usage": {
                        "input_tokens": 10000,
                        "input_tokens_details": {"cached_tokens": 8000},
                        "output_tokens": 500,
                        "output_tokens_details": {"reasoning_tokens": 200},
                    },
                },
            },
        ]
        stream = b"".join(f"data: {json.dumps(ev)}\n\n".encode("utf-8") for ev in events)
        sniffer.feed(stream[:45], 10.2)
        sniffer.feed(stream[45:], 11.5)
        tel = sniffer.finalize(11.6)
        self.assertIsNotNone(sniffer.first_token_mono)
        self.assertEqual(tel["response_id"], "resp_999")
        self.assertEqual(tel["input_tokens"], 10000)
        self.assertEqual(tel["cached_tokens"], 8000)
        self.assertEqual(tel["uncached_input_tokens"], 2000)
        self.assertEqual(tel["output_tokens"], 500)
        self.assertEqual(tel["reasoning_tokens"], 200)
        self.assertEqual(tel["tool_calls_emitted"], 2)


class RunEvalTests(unittest.TestCase):
    def test_build_job_config_terra_medium_custom_mizar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mizar_bin = Path(tmp) / "mizar"
            mizar_bin.write_text("#!/bin/sh\necho 0.1.6\n", encoding="utf-8")
            args = run_eval.parse_args(
                [
                    "--agents",
                    "mizar",
                    "pi",
                    "codex",
                    "--mizar-bin",
                    str(mizar_bin),
                    "--mizar-label",
                    "mizar-dev",
                    "--mizar-version",
                    "0.1.6",
                    "--mizar-args",
                    "--max-steps 50",
                    "--no-upload-rg",
                    "--base-url",
                    "https://api.example.com/v1",
                    "--api-key",
                    "sk-test",
                    "--job-name",
                    "test-job",
                ]
            )
            self.assertEqual(args.effort, "medium")
            self.assertEqual(args.model, "isok/gpt-6.1-terra")

            cfg = run_eval.build_job_config(args, base_url="http://172.17.0.1:3426/v1", api_key="sk-test")
            self.assertEqual(len(cfg["agents"]), 3)
            m_cfg, p_cfg, c_cfg = cfg["agents"]

            self.assertEqual(m_cfg["kwargs"]["harness"], "mizar-dev")
            self.assertEqual(m_cfg["kwargs"]["version"], "0.1.6")
            self.assertEqual(m_cfg["env"]["OPENAI_BASE_URL"], "http://172.17.0.1:3426/a/mizar-dev/v1")
            self.assertIn("--max-steps 50", m_cfg["kwargs"]["run"][-1])
            self.assertIn("--reasoning-effort medium", m_cfg["kwargs"]["run"][-1])
            self.assertEqual(list(m_cfg["kwargs"]["uploads"].values()), ["/tmp/mizar-linux-amd64"])

            self.assertEqual(p_cfg["kwargs"]["thinking"], "medium")
            self.assertEqual(p_cfg["env"]["OPENAI_BASE_URL"], "http://172.17.0.1:3426/a/pi/v1")

            self.assertEqual(c_cfg["kwargs"]["reasoning_effort"], "medium")
            self.assertEqual(c_cfg["kwargs"]["config"]["model_provider"], "study_proxy")
            self.assertEqual(
                c_cfg["kwargs"]["config"]["model_providers"]["study_proxy"]["base_url"],
                "http://172.17.0.1:3426/a/codex/v1",
            )


class ScoreReportTests(unittest.TestCase):
    def _write_trial(
        self,
        job_dir: Path,
        trial_name: str,
        task_name: str,
        harness: str,
        reward: float,
        start: str,
        end: str,
        proxy_trial_id: str | None = None,
        session_id: str | None = None,
    ) -> None:
        tdir = job_dir / trial_name
        adir = tdir / "agent"
        adir.mkdir(parents=True, exist_ok=True)
        res = {
            "task_name": task_name,
            "agent_info": {"name": harness, "version": "1.0", "model_info": {"name": "isok/gpt-6.1-terra"}},
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": None,
            "started_at": start,
            "finished_at": end,
            "agent_execution": {"started_at": start, "finished_at": end},
            "verifier": {"started_at": end, "finished_at": end},
        }
        (tdir / "result.json").write_text(json.dumps(res), encoding="utf-8")
        if proxy_trial_id:
            (adir / "proxy_trial_id.txt").write_text(proxy_trial_id + "\n", encoding="utf-8")
            (adir / "stdout.log").write_text(
                json.dumps({"steps": 4, "tool_calls": 6, "llm_ms": 10000, "tool_ms": 2000, "reason": "completed"}) + "\n",
                encoding="utf-8",
            )
        if session_id:
            traj = {
                "session_id": session_id,
                "steps": [
                    {
                        "source": "agent",
                        "tool_calls": [{"function_name": "bash", "arguments": {"command": "pytest"}}],
                        "observation": {"results": [{"extra": {"is_error": False}}]},
                    }
                ],
            }
            (adir / "trajectory.json").write_text(json.dumps(traj), encoding="utf-8")

    def test_concurrent_proxy_attribution_cache_breaks_and_compare(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            job = root / "concurrent-compare"
            job.mkdir()
            (job / "config.json").write_text(
                json.dumps(
                    {
                        "n_attempts": 1,
                        "datasets": [{"task_names": ["*/mvcc-lsm-compaction", "*/bun-sourcemap-leak"]}],
                        "agents": [{"kwargs": {"harness": "mizar"}}, {"name": "pi"}],
                    }
                ),
                encoding="utf-8",
            )
            # Overlapping timestamps intentionally test concurrent attribution!
            self._write_trial(
                job,
                "mvcc-lsm-compaction__1",
                "terminal-bench/mvcc-lsm-compaction",
                "mizar",
                1.0,
                "2026-11-22T01:00:00Z",
                "2026-11-22T01:05:00Z",
                proxy_trial_id="m-trial-1",
            )
            self._write_trial(
                job,
                "bun-sourcemap-leak__2",
                "terminal-bench/bun-sourcemap-leak",
                "mizar",
                0.0,
                "2026-11-22T01:00:00Z",
                "2026-11-22T01:05:00Z",
                proxy_trial_id="m-trial-2",
            )
            self._write_trial(
                job,
                "mvcc-lsm-compaction__3",
                "terminal-bench/mvcc-lsm-compaction",
                "pi",
                0.0,
                "2026-11-22T01:00:00Z",
                "2026-11-22T01:05:00Z",
                session_id="pi-sess-1",
            )
            self._write_trial(
                job,
                "bun-sourcemap-leak__4",
                "terminal-bench/bun-sourcemap-leak",
                "pi",
                1.0,
                "2026-11-22T01:00:00Z",
                "2026-11-22T01:05:00Z",
                session_id="pi-sess-2",
            )

            proxy_records = [
                # Mizar trial 1: turn 1 cold (0 cached), turn 2 warm (8000 cached), turn 3 break (0 cached)
                {
                    "ts": "2026-11-22T01:01:00Z",
                    "harness": "mizar",
                    "trial_id": "m-trial-1",
                    "status": 200,
                    "input_tokens": 5000,
                    "cached_tokens": 0,
                    "uncached_input_tokens": 5000,
                    "output_tokens": 100,
                    "reasoning_tokens": 50,
                    "prompt_cache_key": "k1",
                    "ttft_ms": 500,
                    "duration_ms": 2000,
                },
                {
                    "ts": "2026-11-22T01:02:00Z",
                    "harness": "mizar",
                    "trial_id": "m-trial-1",
                    "status": 200,
                    "input_tokens": 10000,
                    "cached_tokens": 8000,
                    "uncached_input_tokens": 2000,
                    "output_tokens": 200,
                    "reasoning_tokens": 100,
                    "prompt_cache_key": "k1",
                    "ttft_ms": 400,
                    "duration_ms": 2500,
                },
                {
                    "ts": "2026-11-22T01:03:00Z",
                    "harness": "mizar",
                    "trial_id": "m-trial-1",
                    "status": 200,
                    "input_tokens": 12000,
                    "cached_tokens": 0,
                    "uncached_input_tokens": 12000,
                    "output_tokens": 100,
                    "reasoning_tokens": 50,
                    "prompt_cache_key": "k1",
                    "ttft_ms": 600,
                    "duration_ms": 2000,
                },
                # Mizar trial 2
                {
                    "ts": "2026-11-22T01:01:30Z",
                    "harness": "mizar",
                    "trial_id": "m-trial-2",
                    "status": 200,
                    "input_tokens": 4000,
                    "cached_tokens": 0,
                    "uncached_input_tokens": 4000,
                    "output_tokens": 100,
                    "reasoning_tokens": 20,
                    "prompt_cache_key": None,
                    "ttft_ms": 450,
                    "duration_ms": 1800,
                },
                # Pi trials matched by session_id == prompt_cache_key despite identical timestamps
                {
                    "ts": "2026-11-22T01:01:10Z",
                    "harness": "pi",
                    "status": 200,
                    "input_tokens": 6000,
                    "cached_tokens": 4000,
                    "uncached_input_tokens": 2000,
                    "output_tokens": 100,
                    "reasoning_tokens": 30,
                    "prompt_cache_key": "pi-sess-1",
                    "ttft_ms": 300,
                    "duration_ms": 1500,
                },
                {
                    "ts": "2026-11-22T01:01:20Z",
                    "harness": "pi",
                    "status": 200,
                    "input_tokens": 8000,
                    "cached_tokens": 7000,
                    "uncached_input_tokens": 1000,
                    "output_tokens": 100,
                    "reasoning_tokens": 20,
                    "prompt_cache_key": "pi-sess-2",
                    "ttft_ms": 250,
                    "duration_ms": 1200,
                },
            ]
            (job / "proxy_usage.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in proxy_records), encoding="utf-8"
            )

            manifest = score_report.load_manifest_tasks()
            summary = score_report.summarize_jobs(
                [job], pricing=score_report.DEFAULT_PRICING, pricing_source="terra_default", manifest_tasks=manifest
            )
            self.assertEqual(summary["counts"]["scored"], 4)
            self.assertEqual(summary["usage"]["attributed_trial_calls"], 6)
            self.assertEqual(summary["quadrant"]["primary_only_pass"], 1)
            self.assertEqual(summary["quadrant"]["primary_only_fail"], 1)

            by_h = {s["harness"]: s for s in summary["standings"]}
            m_st = by_h["mizar"]
            self.assertEqual(m_st["usage"]["cache_breaks"], 1)
            self.assertAlmostEqual(m_st["usage"]["warmup_cache_hit_rate"], 8000 / 22000, places=5)
            self.assertEqual(m_st["usage"]["calls"], 4)
            # Check exact Terra cost: (23000 * 1.0 + 8000 * 0.05 + 500 * 5.0) / 1e6 = 0.0259
            self.assertAlmostEqual(m_st["total_cost_usd"], 0.0259, places=6)

            md = score_report.render_markdown(summary)
            html_out = score_report.render_html(summary)
            self.assertIn("Warmup hit", md)
            self.assertIn("FrontierHarness Standings", html_out)

    def test_live_http_sse_proxy_roundtrip(self) -> None:
        import threading
        import urllib.request
        from http.server import BaseHTTPRequestHandler, HTTPServer

        received: list[tuple[str, dict[str, object]]] = []

        class MockUpstream(BaseHTTPRequestHandler):
            def log_message(self, format: str, *args: object) -> None:
                return

            def do_POST(self) -> None:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                received.append((self.path, body))
                events = [
                    {"type": "response.output_text.delta", "delta": "ok"},
                    {
                        "type": "response.completed",
                        "response": {
                            "id": "resp_live_1",
                            "status": "completed",
                            "output": [{"type": "function_call", "name": "bash"}],
                            "usage": {
                                "input_tokens": 2000,
                                "input_tokens_details": {"cached_tokens": 1500},
                                "output_tokens": 100,
                                "output_tokens_details": {"reasoning_tokens": 40},
                            },
                        },
                    },
                ]
                payload = b"".join(f"data: {json.dumps(e)}\n\n".encode("utf-8") for e in events)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                self.wfile.write(payload)

        with tempfile.TemporaryDirectory() as tmp:
            usage_log = Path(tmp) / "proxy_usage.jsonl"
            upstream = HTTPServer(("127.0.0.1", 0), MockUpstream)
            up_port = upstream.server_address[1]
            up_thread = threading.Thread(target=upstream.serve_forever, daemon=True)
            up_thread.start()

            proxy = eval_proxy.StudyProxyServer(
                ("127.0.0.1", 0),
                upstream_url=f"http://127.0.0.1:{up_port}/v1",
                effort="medium",
                max_tokens=131072,
                force_model="isok/gpt-6.1-terra",
                usage_log_path=str(usage_log),
            )
            px_port = proxy.server_address[1]
            px_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
            px_thread.start()
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{px_port}/a/mizar/t/trial-e2e/v1/responses",
                    data=json.dumps({"model": "ignored", "input": "hello", "reasoning": {"effort": "max"}}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:
                    sse_body = resp.read().decode("utf-8")
                    self.assertEqual(resp.status, 200)
                    self.assertIn("resp_live_1", sse_body)
            finally:
                proxy.shutdown()
                proxy.server_close()
                upstream.shutdown()
                upstream.server_close()

            self.assertEqual(len(received), 1)
            self.assertEqual(received[0][0], "/v1/responses")
            self.assertEqual(received[0][1]["model"], "isok/gpt-6.1-terra")
            self.assertEqual(received[0][1]["reasoning"], {"effort": "medium"})
            self.assertEqual(received[0][1]["max_output_tokens"], 131072)

            lines = [json.loads(line) for line in usage_log.read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(lines), 1)
            self.assertEqual(lines[0]["harness"], "mizar")
            self.assertEqual(lines[0]["trial_id"], "trial-e2e")
            self.assertEqual(lines[0]["cached_tokens"], 1500)
            self.assertEqual(lines[0]["uncached_input_tokens"], 500)
            self.assertEqual(lines[0]["tool_calls_emitted"], 1)


if __name__ == "__main__":
    unittest.main()
