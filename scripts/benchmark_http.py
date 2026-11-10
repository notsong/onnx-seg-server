#!/usr/bin/env python3
"""Simple external HTTP load test; requires requests. Reports true client end-to-end latency."""
import argparse
import concurrent.futures
import statistics
import time
import requests

def percentile(values, q):
    values = sorted(values)
    index = (len(values) - 1) * q
    a = int(index)
    b = min(a + 1, len(values) - 1)
    return values[a] * (b - index) + values[b] * (index - a)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:9600/infer")
    p.add_argument("--image", required=True)
    p.add_argument("--category", default="1001")
    p.add_argument("--item", default="4")
    p.add_argument("--requests", type=int, default=100)
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--timeout", type=float, default=120)
    a = p.parse_args()
    if a.requests < 1 or a.concurrency < 1 or a.warmup < 0:
        p.error("requests/concurrency must be positive, warmup nonnegative")
    payload = open(a.image, "rb").read()
    def once(_):
        start = time.perf_counter()
        res = requests.post(a.url, data={"catagory": a.category, "item": a.item},
                            files={"file": ("benchmark.png", payload, "image/png")},
                            timeout=a.timeout)
        res.raise_for_status()
        if res.headers.get("Content-Type", "").split(";")[0] != "image/png":
            raise RuntimeError("Server did not respond with PNG")
        return (1000 * (time.perf_counter() - start),
                float(res.headers.get("X-Inference-Ms", "nan")),
                float(res.headers.get("X-Total-Ms", "nan")),
                len(res.content))
    for i in range(a.warmup):
        once(i)
    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
        results = list(pool.map(once, range(a.requests)))
    seconds = time.perf_counter() - start
    latency = [x[0] for x in results]
    server = [x[2] for x in results]
    infer = [x[1] for x in results]
    print(f"Requests={a.requests} concurrency={a.concurrency} elapsed={seconds:.3f}s")
    print(f"Throughput={a.requests/seconds:.2f} req/s")
    for name, values in [("RTT", latency), ("server", server), ("inference", infer)]:
        print(f"{name}: p50={percentile(values, 0.5):.2f}ms p95={percentile(values, 0.95):.2f}ms p99={percentile(values, 0.99):.2f}ms avg={statistics.mean(values):.2f}ms")
    print(f"Response size: {results[0][3]} bytes")
if __name__ == "__main__":
    main()
