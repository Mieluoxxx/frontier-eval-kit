#!/usr/bin/env python3
"""Lightweight HTTP/SSE study proxy enforcing uniform effort/tokens and logging telemetry."""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import signal
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit, urlunsplit

SIDE_CALL_THRESHOLD = 1024
HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailers",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
ROUTE_PREFIX_RE = re.compile(r"^/a/([^/]+)(?:/t/([^/]+))?(/v1(?:/.*)?)$")


def extract_route_context(path: str) -> tuple[str | None, str | None, str]:
    """Extract optional (/a/<harness>[/t/<trial_id>]) routing tags and return normalized path."""
    split = urlsplit(path)
    match = ROUTE_PREFIX_RE.match(split.path)
    if not match:
        return None, None, path
    harness, trial_id, remainder = match.groups()
    normalized = urlunsplit(("", "", remainder, split.query, split.fragment))
    return harness, trial_id, normalized


def _extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("content")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_first_user_prompt_hash(payload: dict[str, Any]) -> str | None:
    raw_input = payload.get("input")
    if isinstance(raw_input, str) and raw_input.strip():
        return hashlib.sha256(raw_input.strip().encode("utf-8")).hexdigest()[:16]

    messages = raw_input if isinstance(raw_input, list) else payload.get("messages")
    if isinstance(messages, list):
        for item in messages:
            if not isinstance(item, dict):
                continue
            if item.get("role") == "user":
                text = _extract_text_from_content(item.get("content")).strip()
                if text:
                    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return None


def inspect_and_rewrite_payload(
    path: str,
    raw_body: bytes,
    effort: str,
    max_tokens: int,
    force_model: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Enforce model/effort/output cap and return (rewritten_bytes, request_metadata)."""
    meta: dict[str, Any] = {
        "endpoint": urlsplit(path).path.rstrip("/"),
        "request_bytes": len(raw_body),
    }
    if not raw_body:
        return raw_body, meta
    try:
        payload: Any = json.loads(raw_body)
    except json.JSONDecodeError:
        return raw_body, meta
    if not isinstance(payload, dict):
        return raw_body, meta

    clean_path = meta["endpoint"]
    meta["requested_model"] = payload.get("model")
    meta["prompt_cache_key"] = payload.get("prompt_cache_key")
    meta["previous_response_id"] = payload.get("previous_response_id")
    meta["user_prompt_sha256"] = _extract_first_user_prompt_hash(payload)

    items = payload.get("input") if isinstance(payload.get("input"), list) else payload.get("messages")
    if isinstance(items, list):
        meta["input_item_count"] = len(items)
    tools = payload.get("tools")
    if isinstance(tools, list):
        meta["tool_count"] = len(tools)

    is_llm_endpoint = clean_path.endswith("/responses") or clean_path.endswith("/chat/completions")
    if is_llm_endpoint and force_model:
        payload["model"] = force_model
    meta["model"] = payload.get("model")

    if clean_path.endswith("/responses"):
        requested = payload.get("max_output_tokens")
        is_side_call = isinstance(requested, int) and requested < SIDE_CALL_THRESHOLD
        if not is_side_call:
            payload["max_output_tokens"] = max_tokens
        if effort == "off":
            payload.pop("reasoning", None)
        else:
            reasoning = payload.get("reasoning")
            if not isinstance(reasoning, dict):
                reasoning = {}
            reasoning["effort"] = effort
            payload["reasoning"] = reasoning
        meta["is_side_call"] = is_side_call
        meta["max_output_tokens"] = payload.get("max_output_tokens")
        meta["effort"] = effort
    elif clean_path.endswith("/chat/completions"):
        requested = payload.get("max_completion_tokens") or payload.get("max_tokens")
        is_side_call = isinstance(requested, int) and requested < SIDE_CALL_THRESHOLD
        if not is_side_call:
            payload.pop("max_completion_tokens", None)
            payload["max_tokens"] = max_tokens
        if effort == "off":
            payload.pop("reasoning_effort", None)
        else:
            payload["reasoning_effort"] = effort
        meta["is_side_call"] = is_side_call
        meta["max_output_tokens"] = payload.get("max_completion_tokens") or payload.get("max_tokens")
        meta["effort"] = effort

    rewritten = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    meta["request_bytes"] = len(rewritten)
    return rewritten, meta


def rewrite_payload(
    path: str,
    raw_body: bytes,
    effort: str,
    max_tokens: int,
    force_model: str | None = None,
) -> bytes:
    """Backward-compatible helper returning only the rewritten JSON payload."""
    rewritten, _ = inspect_and_rewrite_payload(
        path, raw_body, effort=effort, max_tokens=max_tokens, force_model=force_model
    )
    return rewritten


def extract_usage_telemetry(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract normalized token counts, tool calls, response_id, and optional cost."""
    out: dict[str, Any] = {}
    resp_obj = payload.get("response") if isinstance(payload.get("response"), dict) else payload
    if isinstance(resp_obj.get("id"), str):
        out["response_id"] = resp_obj["id"]

    status = resp_obj.get("status")
    if isinstance(status, str):
        out["stop_reason"] = status
    if isinstance(resp_obj.get("incomplete_details"), dict):
        reason = resp_obj["incomplete_details"].get("reason")
        if reason:
            out["stop_reason"] = f"incomplete:{reason}"

    choices = resp_obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0] if isinstance(choices[0], dict) else {}
        if isinstance(first.get("finish_reason"), str):
            out["stop_reason"] = first["finish_reason"]
        msg = first.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("tool_calls"), list):
            names = [
                (tc.get("function") or {}).get("name")
                for tc in msg["tool_calls"]
                if isinstance(tc, dict)
            ]
            out["tool_calls_emitted"] = len(msg["tool_calls"])
            out["tool_names"] = [n for n in names if isinstance(n, str)]

    output_items = resp_obj.get("output")
    if isinstance(output_items, list):
        tool_names: list[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            itype = str(item.get("type") or "")
            if itype.endswith("_call") or itype in {"function_call", "custom_tool_call", "local_shell_call"}:
                name = item.get("name") or itype
                tool_names.append(str(name))
        out["tool_calls_emitted"] = len(tool_names)
        if tool_names:
            out["tool_names"] = tool_names

    usage = resp_obj.get("usage")
    if isinstance(usage, dict):
        if "input_tokens" in usage:
            raw_in = int(usage.get("input_tokens") or 0)
            in_details = usage.get("input_tokens_details") or {}
            cached = int(
                in_details.get("cached_tokens")
                or usage.get("cache_read_input_tokens")
                or 0
            )
            cwrite = int(usage.get("cache_creation_input_tokens") or 0)
            out_tok = int(usage.get("output_tokens") or 0)
            out_details = usage.get("output_tokens_details") or {}
            reason_tok = int(out_details.get("reasoning_tokens") or 0)
            if "cache_read_input_tokens" in usage and not in_details.get("cached_tokens"):
                uncached = raw_in
                total_in = raw_in + cached
            else:
                total_in = raw_in
                uncached = max(0, raw_in - cached)
        else:
            raw_in = int(usage.get("prompt_tokens") or 0)
            in_details = usage.get("prompt_tokens_details") or {}
            cached = int(in_details.get("cached_tokens") or 0)
            cwrite = int(in_details.get("cache_creation_tokens") or 0)
            out_tok = int(usage.get("completion_tokens") or 0)
            out_details = usage.get("completion_tokens_details") or {}
            reason_tok = int(out_details.get("reasoning_tokens") or 0)
            total_in = raw_in
            uncached = max(0, raw_in - cached)

        out.update(
            {
                "input_tokens": total_in,
                "cached_tokens": cached,
                "uncached_input_tokens": uncached,
                "cache_write_tokens": cwrite,
                "output_tokens": out_tok,
                "reasoning_tokens": reason_tok,
            }
        )
        cost = usage.get("cost") if usage.get("cost") is not None else usage.get("cost_usd")
        if isinstance(cost, (int, float)):
            out["upstream_cost_usd"] = float(cost)

    err = resp_obj.get("error")
    if isinstance(err, dict):
        msg = err.get("message") or err.get("code") or err.get("type")
        if msg:
            out["error"] = str(msg)[:300]
    return out


class SSETelemetrySniffer:
    """Incrementally inspects SSE chunks without buffering or stalling upstream bytes."""

    def __init__(self, start_mono: float) -> None:
        self._start_mono = start_mono
        self._buf = b""
        self.first_token_mono: float | None = None
        self.telemetry: dict[str, Any] = {}
        self._chat_tool_indices: dict[int, str] = {}
        self._responses_tool_ids: dict[str, str] = {}

    def feed(self, chunk: bytes, now_mono: float) -> None:
        self._buf += chunk
        while b"\n" in self._buf:
            raw_line, self._buf = self._buf.split(b"\n", 1)
            self._handle_line(raw_line.decode("utf-8", errors="replace").strip(), now_mono)

    def finalize(self, now_mono: float) -> dict[str, Any]:
        if self._buf.strip():
            self._handle_line(self._buf.decode("utf-8", errors="replace").strip(), now_mono)
            self._buf = b""
        if "tool_calls_emitted" not in self.telemetry:
            if self._responses_tool_ids:
                names = [n for n in self._responses_tool_ids.values() if n]
                self.telemetry["tool_calls_emitted"] = len(self._responses_tool_ids)
                if names:
                    self.telemetry["tool_names"] = names
            elif self._chat_tool_indices:
                names = [n for _, n in sorted(self._chat_tool_indices.items()) if n]
                self.telemetry["tool_calls_emitted"] = len(self._chat_tool_indices)
                if names:
                    self.telemetry["tool_names"] = names
        return self.telemetry

    def _handle_line(self, line: str, now_mono: float) -> None:
        if not line.startswith("data:"):
            return
        body = line[5:].strip()
        if not body or body == "[DONE]":
            return
        try:
            event = json.loads(body)
        except json.JSONDecodeError:
            return
        if not isinstance(event, dict):
            return

        etype = event.get("type")
        if isinstance(etype, str) and (
            etype.endswith(".delta") or etype == "response.output_item.added"
        ):
            if self.first_token_mono is None:
                self.first_token_mono = now_mono

        if etype == "response.output_item.added" and isinstance(event.get("item"), dict):
            item = event["item"]
            itype = str(item.get("type") or "")
            if itype.endswith("_call") or itype in {"function_call", "custom_tool_call", "local_shell_call"}:
                call_id = str(item.get("id") or item.get("call_id") or len(self._responses_tool_ids))
                self._responses_tool_ids[call_id] = str(item.get("name") or itype)

        choices = event.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            delta = choices[0].get("delta")
            if isinstance(delta, dict) and (
                delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
            ):
                if self.first_token_mono is None:
                    self.first_token_mono = now_mono
                for tc in delta.get("tool_calls") or []:
                    if isinstance(tc, dict):
                        idx = int(tc.get("index") or 0)
                        fn_name = (tc.get("function") or {}).get("name")
                        if fn_name or idx not in self._chat_tool_indices:
                            self._chat_tool_indices[idx] = str(fn_name or self._chat_tool_indices.get(idx) or "")
            if choices[0].get("finish_reason"):
                self.telemetry["stop_reason"] = str(choices[0]["finish_reason"])

        extracted = extract_usage_telemetry(event)
        if extracted:
            self.telemetry.update({k: v for k, v in extracted.items() if v is not None})


class StudyProxyServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        upstream_url: str,
        effort: str,
        max_tokens: int,
        force_model: str | None = None,
        usage_log_path: str | None = None,
    ) -> None:
        parsed = urlsplit(upstream_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Invalid upstream URL: {upstream_url}")
        self.upstream_scheme = parsed.scheme
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.upstream_base_path = parsed.path.rstrip("/")
        self.effort = effort
        self.max_tokens = max_tokens
        self.force_model = force_model
        self.usage_log_path = usage_log_path
        self._log_lock = threading.Lock()
        super().__init__(server_address, StudyProxyHandler)

    def record_usage(self, record: dict[str, Any]) -> None:
        if not self.usage_log_path:
            return
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._log_lock:
            parent = os.path.dirname(self.usage_log_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self.usage_log_path, "a", encoding="utf-8") as fh:
                fh.write(line)


class StudyProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _forward(self) -> None:
        srv: StudyProxyServer = self.server  # type: ignore[assignment]
        harness, trial_id, clean_path = extract_route_context(self.path)

        length = int(self.headers.get("Content-Length") or 0)
        raw_body = self.rfile.read(length) if length > 0 else b""
        req_meta: dict[str, Any] = {"endpoint": urlsplit(clean_path).path.rstrip("/")}
        if self.command == "POST" and "json" in (self.headers.get("Content-Type") or "").lower():
            raw_body, req_meta = inspect_and_rewrite_payload(
                clean_path, raw_body, srv.effort, srv.max_tokens, force_model=srv.force_model
            )

        target_path = clean_path
        if srv.upstream_base_path and target_path.startswith("/v1/"):
            # Upstream base already ends with /v1 when configured as http://host:port/v1.
            if srv.upstream_base_path.endswith("/v1"):
                target_path = srv.upstream_base_path + target_path[3:]
            else:
                target_path = srv.upstream_base_path + target_path
        elif srv.upstream_base_path:
            target_path = srv.upstream_base_path + target_path

        fwd_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP_HEADERS
        }
        default_port = 443 if srv.upstream_scheme == "https" else 80
        fwd_headers["Host"] = (
            srv.upstream_host
            if srv.upstream_port == default_port
            else f"{srv.upstream_host}:{srv.upstream_port}"
        )
        if raw_body or self.command in {"POST", "PUT", "PATCH"}:
            fwd_headers["Content-Length"] = str(len(raw_body))

        conn_cls = (
            http.client.HTTPSConnection
            if srv.upstream_scheme == "https"
            else http.client.HTTPConnection
        )
        conn = conn_cls(srv.upstream_host, srv.upstream_port, timeout=1800)
        started_ts = datetime.now(timezone.utc).isoformat()
        start_mono = time.monotonic()
        resp_telemetry: dict[str, Any] = {}
        resp_status = 502
        first_byte_mono: float | None = None

        try:
            conn.request(self.command, target_path, body=raw_body if raw_body else None, headers=fwd_headers)
            resp = conn.getresponse()
            resp_status = resp.status

            self.send_response(resp.status, resp.reason)
            for k, v in resp.getheaders():
                if k.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(k, v)

            content_len = resp.getheader("Content-Length")
            content_type = (resp.getheader("Content-Type") or "").lower()
            use_chunked = content_len is None or "text/event-stream" in content_type
            is_sse = "text/event-stream" in content_type

            if use_chunked:
                self.send_header("Transfer-Encoding", "chunked")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                sniffer = SSETelemetrySniffer(start_mono) if is_sse else None
                body_chunks: list[bytes] = []
                while True:
                    # read1 yields immediately per socket frame so SSE events never stall.
                    chunk = resp.read1(8192)
                    if not chunk:
                        break
                    now_mono = time.monotonic()
                    if first_byte_mono is None:
                        first_byte_mono = now_mono
                    self.wfile.write(f"{len(chunk):x}\r\n".encode("ascii") + chunk + b"\r\n")
                    self.wfile.flush()
                    if sniffer is not None:
                        sniffer.feed(chunk, now_mono)
                    elif sum(len(c) for c in body_chunks) < 262144:
                        body_chunks.append(chunk)
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
                if sniffer is not None:
                    resp_telemetry = sniffer.finalize(time.monotonic())
                    if sniffer.first_token_mono is not None:
                        first_byte_mono = sniffer.first_token_mono
                elif body_chunks and "json" in content_type:
                    try:
                        parsed_body = json.loads(b"".join(body_chunks))
                        if isinstance(parsed_body, dict):
                            resp_telemetry = extract_usage_telemetry(parsed_body)
                    except json.JSONDecodeError:
                        pass
            else:
                body = resp.read()
                first_byte_mono = time.monotonic()
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if body:
                    self.wfile.write(body)
                self.wfile.flush()
                if body and ("json" in content_type or body.lstrip().startswith(b"{")):
                    try:
                        parsed_body = json.loads(body)
                        if isinstance(parsed_body, dict):
                            resp_telemetry = extract_usage_telemetry(parsed_body)
                    except json.JSONDecodeError:
                        pass
        except Exception as exc:
            resp_telemetry["error"] = f"Study proxy upstream error: {exc}"
            msg = json.dumps({"error": {"message": resp_telemetry["error"]}}).encode("utf-8")
            try:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(msg)))
                self.end_headers()
                self.wfile.write(msg)
                self.wfile.flush()
            except OSError:
                pass
        finally:
            conn.close()
            end_mono = time.monotonic()
            endpoint = str(req_meta.get("endpoint") or "")
            if self.command == "POST" and (
                endpoint.endswith("/responses") or endpoint.endswith("/chat/completions")
            ):
                ttft_ms = (
                    int(round((first_byte_mono - start_mono) * 1000))
                    if first_byte_mono is not None
                    else None
                )
                duration_ms = int(round((end_mono - start_mono) * 1000))
                record = {
                    "ts": started_ts,
                    "harness": harness,
                    "trial_id": trial_id,
                    "status": resp_status,
                    "ttft_ms": ttft_ms,
                    "duration_ms": duration_ms,
                    **req_meta,
                    **resp_telemetry,
                }
                srv.record_usage(record)

    def do_GET(self) -> None:
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def do_DELETE(self) -> None:
        self._forward()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=3426)
    parser.add_argument("--upstream", required=True, help="Upstream OpenAI-compatible base URL")
    parser.add_argument("--effort", default="medium")
    parser.add_argument("--max-tokens", type=int, default=131072)
    parser.add_argument("--force-model", help="Optional model ID override written into LLM payloads")
    parser.add_argument("--usage-log", help="Path to append JSONL per-call telemetry records")
    parser.add_argument("--pid-file")
    args = parser.parse_args()

    server = StudyProxyServer(
        (args.listen_host, args.listen_port),
        upstream_url=args.upstream,
        effort=args.effort,
        max_tokens=args.max_tokens,
        force_model=args.force_model,
        usage_log_path=args.usage_log,
    )
    if args.pid_file:
        with open(args.pid_file, "w", encoding="utf-8") as fh:
            fh.write(f"{os.getpid()}\n")

    def _stop(_sig: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _stop)
    actual_port = server.server_address[1]
    print(
        f"Study proxy listening on {args.listen_host}:{actual_port} -> {args.upstream} "
        f"(effort={args.effort}, max_tokens={args.max_tokens})",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
