#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_API_KEY = "dev-key-change-in-production"


class SmokeFailure(RuntimeError):
    """Raised when the production smoke runner finds a blocking failure."""

    def __init__(self, message: str, summary: dict[str, Any] | None = None):
        super().__init__(message)
        self.summary = dict(summary or {})


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def _request_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    query: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    timeout: float = 5.0,
) -> tuple[int, dict[str, Any]]:
    if query:
        url = f"{url}?{urllib.parse.urlencode(query)}"

    body = None
    request_headers = dict(headers or {})
    if json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"

    req = urllib.request.Request(url, method=method, headers=request_headers, data=body)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return resp.status, json.loads(payload)


def _tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _terminate_process(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGINT)
    except ProcessLookupError:
        return

    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        proc.wait(timeout=5)


def _wait_for_health(base_url: str, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None

    while time.monotonic() < deadline:
        try:
            status, payload = _request_json("GET", f"{base_url}/health", timeout=5.0)
            if status == 200:
                return payload
            last_error = f"unexpected HTTP status {status}"
        except Exception as exc:  # pragma: no cover - exercised in live smoke only
            last_error = str(exc)
        time.sleep(1.0)

    raise SmokeFailure(f"service did not become healthy in time: {last_error or 'unknown error'}")


def run_smoke(
    *,
    port: int | None,
    home_dir: str | None,
    timeout_seconds: float,
    keep_home: bool,
) -> dict[str, Any]:
    runtime_home = Path(home_dir).expanduser() if home_dir else Path(tempfile.mkdtemp(prefix="mollygraph-production-smoke."))
    runtime_home.mkdir(parents=True, exist_ok=True)
    logs_path = runtime_home / "logs" / "service.log"
    port = int(port or _find_free_port())
    base_url = f"http://127.0.0.1:{port}"
    api_key = os.environ.get("MOLLYGRAPH_API_KEY", DEFAULT_API_KEY)
    proc: subprocess.Popen[str] | None = None

    summary: dict[str, Any] = {
        "status": "failed",
        "port": port,
        "home_dir": str(runtime_home),
        "logs_path": str(logs_path),
        "base_url": base_url,
    }

    try:
        env = os.environ.copy()
        env["GRAPH_MEMORY_PORT"] = str(port)
        env["MOLLYGRAPH_HOME_DIR"] = str(runtime_home)
        env["MOLLYGRAPH_ALLOW_CONCURRENT_LOCAL_RUNS"] = "1"

        proc = subprocess.Popen(
            [str(REPO_ROOT / "scripts" / "start.sh")],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        startup_health = _wait_for_health(base_url, timeout_seconds)
        summary["startup_health"] = startup_health
        if startup_health.get("status") != "healthy":
            raise SmokeFailure(f"startup health was not healthy: {startup_health}")
        if startup_health.get("operator_advisories"):
            raise SmokeFailure(
                "fresh runtime unexpectedly reported operator advisories: "
                + "; ".join(str(v) for v in startup_health["operator_advisories"])
            )

        headers = {"Authorization": f"Bearer {api_key}"}
        person = "Elena Ward"
        org = "Signal Foundry"
        content = f"{person} works at {org}."

        ingest_status, ingest_payload = _request_json(
            "POST",
            f"{base_url}/ingest",
            headers=headers,
            json_body={
                "content": content,
                "source": "session",
                "speaker": person,
                "priority": 1,
            },
            timeout=10.0,
        )
        summary["ingest"] = {"status": ingest_status, "body": ingest_payload}
        if ingest_status != 200:
            raise SmokeFailure(f"ingest failed: {ingest_status} {ingest_payload}")

        deadline = time.monotonic() + timeout_seconds
        health_samples: list[dict[str, Any]] = []
        entity_payload: dict[str, Any] | None = None
        entity_path = urllib.parse.quote(person)

        while time.monotonic() < deadline:
            sample_start = time.perf_counter()
            health_status, health_payload = _request_json(
                "GET",
                f"{base_url}/health",
                timeout=5.0,
            )
            sample_ms = (time.perf_counter() - sample_start) * 1000.0
            if health_status != 200:
                raise SmokeFailure(f"health probe failed during ingest: {health_status}")

            health_samples.append(
                {
                    "status": health_payload.get("status"),
                    "queue_processing": health_payload.get("queue_processing"),
                    "queue_stuck": health_payload.get("queue_stuck"),
                    "worker_status": health_payload.get("worker_status"),
                    "latency_ms": round(sample_ms, 2),
                }
            )

            try:
                entity_status, entity_candidate = _request_json(
                    "GET",
                    f"{base_url}/entity/{entity_path}",
                    headers=headers,
                    timeout=5.0,
                )
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
                entity_status = 404
                entity_candidate = {}

            if entity_status == 200 and any(
                fact.get("rel_type") == "WORKS_AT" and fact.get("target_name") == org
                for fact in entity_candidate.get("facts", [])
            ):
                entity_payload = entity_candidate
                break

            time.sleep(1.0)

        if entity_payload is None:
            raise SmokeFailure(
                "entity lookup did not produce the expected WORKS_AT fact before timeout"
            )

        summary["health_samples"] = health_samples
        summary["entity"] = entity_payload

        query_status, query_payload = _request_json(
            "GET",
            f"{base_url}/query",
            headers=headers,
            query={"q": f"Where does {person} work?"},
            timeout=10.0,
        )
        summary["query"] = {"status": query_status, "body": query_payload}
        if query_status != 200:
            raise SmokeFailure(f"query failed: {query_status} {query_payload}")
        if query_payload.get("result_count", 0) < 1:
            raise SmokeFailure("query returned no results")

        result_entities = [result.get("entity") for result in query_payload.get("results", [])]
        if person not in result_entities:
            raise SmokeFailure(f"query did not return expected entity: {result_entities}")

        summary["status"] = "ok"
        return summary
    except Exception as exc:
        summary["error"] = str(exc)
        summary["log_tail"] = _tail(logs_path)
        if proc is not None and proc.poll() is not None:
            stdout, stderr = proc.communicate()
            summary["start_stdout"] = stdout.strip()
            summary["start_stderr"] = stderr.strip()
        raise SmokeFailure(str(exc), summary) from exc
    finally:
        if proc is not None:
            _terminate_process(proc)
        if not keep_home and home_dir is None:
            shutil.rmtree(runtime_home, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run MollyGraph's isolated production smoke test.")
    parser.add_argument("--port", type=int, default=None, help="Port to run the temporary service on.")
    parser.add_argument("--home-dir", default=None, help="Optional runtime home dir to reuse.")
    parser.add_argument("--timeout", type=float, default=90.0, help="Timeout in seconds for startup and ingest checks.")
    parser.add_argument("--keep-home", action="store_true", help="Keep the temporary runtime directory after the run.")
    parser.add_argument("--json", action="store_true", help="Print the final summary as JSON.")
    args = parser.parse_args()

    try:
        summary = run_smoke(
            port=args.port,
            home_dir=args.home_dir,
            timeout_seconds=args.timeout,
            keep_home=args.keep_home,
        )
    except Exception as exc:
        payload = {"status": "failed", "error": str(exc)}
        if isinstance(exc, SmokeFailure) and exc.summary:
            payload.update(exc.summary)
        if args.json:
            print(json.dumps(payload))
        else:
            print(json.dumps(payload, indent=2))
        return 1

    if args.json:
        print(json.dumps(summary))
    else:
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
