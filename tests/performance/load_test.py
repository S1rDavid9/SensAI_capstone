"""Stress test: many learners using the app at the same moment.

Starts the real server (uvicorn, one worker, like on the pilot day) against
a throwaway database, then simulates N learners who all join and answer all
15 questions as fast as possible, with no thinking time. That is far harsher
than a real class, where answers are 10-60 seconds apart. Measures response
times per request type, failed requests, and the server's CPU and memory.

    uv run python -m tests.performance.load_test            # 10, 30, 60, 100 learners
    uv run python -m tests.performance.load_test --learners 30 --mode adaptive

Results: docs/testing/load_test_results.json and .md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import httpx
import psutil

from pilot.core.tasks import CHECKIN_AFTER, SESSION_LENGTH, QuestionBank

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "testing"
KEY = {"X-Researcher-Key": "load-test-key"}
BANK = QuestionBank()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerUnderTest:
    """Runs the app in its own process so its CPU and memory can be measured."""

    def __init__(self, db_path: Path, workers: int = 1) -> None:
        self.port = free_port()
        self.workers = workers
        env = {**os.environ, "PILOT_DB": str(db_path), "PILOT_RESEARCHER_KEY": KEY["X-Researcher-Key"]}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "pilot.server:app", "--port", str(self.port), "--log-level", "warning",
             "--workers", str(workers)],
            cwd=ROOT, env=env)
        self.url = f"http://127.0.0.1:{self.port}"
        self._wait_until_up()
        self.ps = psutil.Process(self.proc.pid)
        time.sleep(3 if workers > 1 else 0)   # let every worker finish loading the models
        self.samples: list[tuple[float, float]] = []   # (cpu %, rss MB)
        self._stop = threading.Event()

    def _wait_until_up(self) -> None:
        for _ in range(200):
            try:
                httpx.get(self.url + "/api/time", timeout=0.5)
                return
            except httpx.HTTPError:
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    def _processes(self) -> list[psutil.Process]:
        """The server process and its live worker processes."""
        procs = [self.ps]
        for child in self.ps.children(recursive=True):
            try:
                if child.status() != psutil.STATUS_ZOMBIE:
                    procs.append(child)
            except psutil.Error:
                pass
        return procs

    def sample(self, interval: float = 0.1) -> threading.Thread:
        procs = self._processes()
        for p in procs:
            try:
                p.cpu_percent(None)
            except psutil.Error:
                pass

        def run():
            # CPU and memory summed over the main process and any worker processes.
            while not self._stop.is_set():
                cpu = mem = 0.0
                for p in procs:
                    try:
                        cpu += p.cpu_percent(None)
                        mem += p.memory_info().rss / 2**20
                    except psutil.Error:
                        pass
                self.samples.append((cpu, mem))
                time.sleep(interval)
        t = threading.Thread(target=run, daemon=True)
        t.start()
        return t

    def stop_sampling(self) -> None:
        self._stop.set()

    def idle_memory_mb(self) -> float:
        total = 0
        for p in self._processes():
            try:
                total += p.memory_info().rss
            except psutil.Error:
                pass
        return total / 2**20

    def shutdown(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=10)


async def learner(client: httpx.AsyncClient, url: str, code: str, timings: dict, errors: list, correct_rate: float,
                  rng_seed: int) -> None:
    import random
    rng = random.Random(rng_seed)

    async def call(kind: str, path: str, body: dict) -> dict | None:
        t = time.perf_counter()
        try:
            r = await client.post(url + path, json=body)
            timings.setdefault(kind, []).append((time.perf_counter() - t) * 1000)
            if r.status_code != 200:
                errors.append(f"{kind}: {r.status_code}")
                return None
            return r.json()
        except httpx.HTTPError as e:
            errors.append(f"{kind}: {type(e).__name__}")
            return None

    joined = await call("join", "/api/learner/join", {"code": code})
    if not joined:
        return
    auth = {"code": code, "token": joined["token"]}
    for i in range(SESSION_LENGTH):
        cur = await call("question", "/api/learner/question", auth)
        if not cur or cur.get("done"):
            return
        await call("event", "/api/learner/event", {**auth, "type": "question_shown", "question_index": i})
        q = BANK.get(cur["question"]["id"])
        answer = q.answer if rng.random() < correct_rate else "0"
        res = await call("answer", "/api/learner/answer", {**auth, "question_index": i, "answer": answer,
                                                         "response_time_ms": rng.randint(3000, 40000),
                                                         "first_input_ms": rng.randint(500, 20000)})
        if res and i in CHECKIN_AFTER:
            await call("checkin", "/api/learner/checkin", {**auth, "after_question": i, "choice": "good"})
    await call("finish", "/api/learner/finish", auth)


def pct(values: list[float], p: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(p / 100 * (len(values) - 1)))]


async def run_level(n: int, mode: str, workers: int = 1) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        server = ServerUnderTest(Path(tmp) / "load.sqlite3", workers)
        try:
            idle = server.idle_memory_mb()
            async with httpx.AsyncClient(timeout=30, limits=httpx.Limits(max_connections=n + 5)) as client:
                codes = []
                for _ in range(n):
                    r = await client.post(server.url + "/api/researcher/sessions", headers=KEY, json={
                        "guardian_consent": True, "learner_assent": True, "recorded_by": "load", "mode": mode})
                    codes.append(r.json()["participant_code"])
                timings: dict[str, list[float]] = {}
                errors: list[str] = []
                server.sample()
                start = time.perf_counter()
                await asyncio.gather(*[learner(client, server.url, c, timings, errors, 0.7, i)
                                       for i, c in enumerate(codes)])
                elapsed = time.perf_counter() - start
                server.stop_sampling()
                stored = (await client.get(server.url + "/api/researcher/sessions", headers=KEY)).json()
        finally:
            server.shutdown()
    all_ms = [t for v in timings.values() for t in v]
    cpu = [c for c, _ in server.samples] or [0.0]
    mem = [m for _, m in server.samples] or [idle]
    return {
        "learners": n, "mode": mode, "workers": workers,
        "requests": len(all_ms), "failed_requests": len(errors), "error_examples": errors[:5],
        "wall_time_s": round(elapsed, 2), "throughput_req_per_s": round(len(all_ms) / elapsed, 1),
        "latency_ms": {k: {"median": round(statistics.median(v), 1), "p95": round(pct(v, 95), 1),
                           "max": round(max(v), 1), "n": len(v)} for k, v in sorted(timings.items())},
        "latency_all_ms": {"median": round(statistics.median(all_ms), 1), "p95": round(pct(all_ms, 95), 1),
                           "p99": round(pct(all_ms, 99), 1), "max": round(max(all_ms), 1)},
        "server_cpu_percent": {"mean": round(statistics.mean(cpu), 1), "peak": round(max(cpu), 1)},
        "server_memory_mb": {"idle": round(idle, 1), "peak": round(max(mem), 1)},
        "sessions_finished": sum(s["status"] == "finished" for s in stored),
        "answers_stored": sum(s["answered"] for s in stored),
    }


def machine() -> dict:
    return {"os": f"{platform.system()} {platform.mac_ver()[0] or platform.release()}",
            "cpu": platform.processor() or platform.machine(), "cores": psutil.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 2**30, 1), "python": platform.python_version()}


def to_markdown(results: list[dict], env: dict) -> str:
    lines = [
        "# Stress test results",
        "",
        f"Server machine: {env['os']}, {env['cpu']}, {env['cores']} cores, {env['ram_gb']} GB RAM, Python {env['python']}.",
        "One uvicorn worker, SQLite, the DQN loaded in memory. Every simulated learner answers all 15 questions",
        "with no thinking time, so these loads are far heavier than a real class.",
        "",
        "| Learners at once | Mode | Workers | Requests | Failed | Median ms | p95 ms | p99 ms | Answer p95 ms | CPU mean / peak % | Memory idle → peak MB | Sessions finished |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        a = r["latency_ms"].get("answer", {})
        lines.append(
            f"| {r['learners']} | {r['mode']} | {r.get('workers', 1)} | {r['requests']} | {r['failed_requests']} | "
            f"{r['latency_all_ms']['median']} | {r['latency_all_ms']['p95']} | {r['latency_all_ms']['p99']} | "
            f"{a.get('p95', '-')} | {r['server_cpu_percent']['mean']} / {r['server_cpu_percent']['peak']} | "
            f"{r['server_memory_mb']['idle']} → {r['server_memory_mb']['peak']} | {r['sessions_finished']}/{r['learners']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", type=int, nargs="*", default=[10, 30, 60, 100])
    parser.add_argument("--mode", choices=["shadow", "adaptive", "both"], default="both")
    parser.add_argument("--workers", type=int, nargs="*", default=[1])
    args = parser.parse_args()
    modes = ["shadow", "adaptive"] if args.mode == "both" else [args.mode]
    results = []
    for workers in args.workers:
      for mode in modes:
        for n in args.learners:
            r = asyncio.run(run_level(n, mode, workers))
            results.append(r)
            print(f"{n:4d} learners, {mode:8s}, {workers} worker(s): {r['requests']} requests, {r['failed_requests']} failed, "
                  f"median {r['latency_all_ms']['median']} ms, p95 {r['latency_all_ms']['p95']} ms, "
                  f"CPU peak {r['server_cpu_percent']['peak']}%, memory peak {r['server_memory_mb']['peak']} MB, "
                  f"finished {r['sessions_finished']}/{n}")
    env = machine()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "load_test_results.json").write_text(json.dumps({"machine": env, "results": results}, indent=2))
    (OUT / "load_test_results.md").write_text(to_markdown(results, env))


if __name__ == "__main__":
    main()
