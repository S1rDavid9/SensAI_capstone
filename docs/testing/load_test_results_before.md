# Stress test results

Server machine: Darwin 26.6.2, arm, 8 cores, 8.0 GB RAM, Python 3.11.15.
One uvicorn worker, SQLite, the DQN loaded in memory. Every simulated learner answers all 15 questions
with no thinking time, so these loads are far heavier than a real class.

| Learners at once | Mode | Requests | Failed | Median ms | p95 ms | p99 ms | Answer p95 ms | CPU mean / peak % | Memory idle → peak MB | Sessions finished |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | shadow | 500 | 0 | 6.2 | 52.4 | 140.1 | 72.7 | 73.1 / 115.4 | 277.7 → 279.9 | 10/10 |
| 30 | shadow | 1500 | 0 | 42.7 | 369.6 | 606.6 | 391.1 | 31.2 / 90.2 | 238.3 → 219.0 | 30/30 |
| 60 | shadow | 3000 | 0 | 127.2 | 947.4 | 1629.4 | 887.6 | 27.7 / 110.6 | 306.5 → 308.0 | 60/60 |
| 100 | shadow | 5000 | 0 | 189.4 | 903.0 | 1396.2 | 877.6 | 39.8 / 117.6 | 314.4 → 309.2 | 100/100 |
| 10 | adaptive | 500 | 0 | 3.4 | 15.6 | 2037.3 | 16.0 | 79.1 / 99.7 | 288.1 → 407.9 | 10/10 |
| 30 | adaptive | 1500 | 0 | 16.1 | 252.1 | 2143.3 | 321.5 | 65.0 / 110.8 | 304.5 → 400.5 | 30/30 |
| 60 | adaptive | 3000 | 0 | 130.7 | 1609.9 | 2452.5 | 1343.9 | 35.1 / 127.0 | 305.4 → 365.4 | 60/60 |
| 100 | adaptive | 5000 | 0 | 224.2 | 2144.1 | 3878.6 | 1867.0 | 32.6 / 121.7 | 301.7 → 402.9 | 100/100 |
