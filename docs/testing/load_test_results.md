# Stress test results

Server machine: Darwin 26.6.2, arm, 8 cores, 8.0 GB RAM, Python 3.11.15.
One uvicorn worker, SQLite, the DQN loaded in memory. Every simulated learner answers all 15 questions
with no thinking time, so these loads are far heavier than a real class.

| Learners at once | Mode | Workers | Requests | Failed | Median ms | p95 ms | p99 ms | Answer p95 ms | CPU mean / peak % | Memory idle → peak MB | Sessions finished |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | adaptive | 1 | 1500 | 0 | 43.2 | 376.1 | 612.1 | 408.2 | 33.9 / 92.3 | 217.0 → 216.9 | 30/30 |
