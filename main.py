import argparse
import asyncio
import sys
import time
from collections import deque
from typing import Optional

import aiohttp

RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
GREY   = "\033[90m"

def color(text: str, c: str) -> str:
    return f"{c}{text}{RESET}"

def kv_bar(pct: float, width: int = 36) -> str:
    filled = int(pct / 100 * width)
    c = RED if pct >= 90 else YELLOW if pct >= 70 else GREEN
    return color("█" * filled, c) + color("░" * (width - filled), GREY)

def _parse_histogram(lines: list, metric_name: str) -> dict:
    buckets: dict = {}
    total_sum   = 0.0
    total_count = 0.0
    for line in lines:
        if line.startswith("#") or metric_name not in line:
            continue
        if f"{metric_name}_bucket" in line:
            try:
                le  = float(line.split('le="')[1].split('"')[0])
                val = float(line.split()[-1])
                buckets[le] = buckets.get(le, 0.0) + val
            except (IndexError, ValueError):
                pass
        elif f"{metric_name}_sum" in line:
            try:
                total_sum += float(line.split()[-1])
            except ValueError:
                pass
        elif f"{metric_name}_count" in line:
            try:
                total_count += float(line.split()[-1])
            except ValueError:
                pass
    return {"buckets": buckets, "_sum": total_sum, "_count": total_count}

def percentile_from_histogram(hist: dict, p: float) -> Optional[float]:
    """Linear interpolation over cumulative histogram buckets."""
    buckets = hist.get("buckets", {})
    total   = hist.get("_count", 0)
    if not buckets or total == 0:
        return None
    sorted_les = sorted(b for b in buckets if b != float("inf"))
    target = p / 100.0 * total
    prev_le, prev_count = 0.0, 0.0
    for le in sorted_les:
        count = buckets[le]
        if count >= target:
            if count == prev_count:
                return prev_le
            frac = (target - prev_count) / (count - prev_count)
            return prev_le + frac * (le - prev_le)
        prev_le, prev_count = le, count
    return sorted_les[-1]

def _float(line: str) -> float:
    try:
        return float(line.split()[-1])
    except ValueError:
        return 0.0

def scrape_raw(text: str) -> dict:
    lines = text.splitlines()

    result = {
        "gpu_cache_pct":           0.0,
        "cpu_cache_pct":           0.0,
        "num_running":             0,
        "num_waiting":             0,
        "num_swapped":             0,
        "prompt_tokens_total":     0.0,
        "generation_tokens_total": 0.0,
        "requests_total":          0.0,
        "hist_ttft": {},
        "hist_itl":  {},
        "hist_e2e":  {},
    }

    for line in lines:
        if line.startswith("#"):
            continue

        # KV cache
        if any(k in line for k in ("vllm:gpu_cache_usage_perc",
                                    "gpu_cache_usage_percentage",
                                    "vllm:kv_cache_usage_perc")):
            val = _float(line)
            result["gpu_cache_pct"] = val if val > 1 else val * 100

        elif "vllm:cpu_cache_usage_perc" in line:
            val = _float(line)
            result["cpu_cache_pct"] = val if val > 1 else val * 100

        # Queue
        elif "vllm:num_requests_running" in line:
            result["num_running"] = int(_float(line))
        elif "vllm:num_requests_waiting" in line:
            result["num_waiting"] = int(_float(line))
        elif "vllm:num_requests_swapped" in line:
            result["num_swapped"] = int(_float(line))

        # Token counters – handle naming differences across vLLM versions
        elif "vllm:prompt_tokens_total" in line :
            result["prompt_tokens_total"] = _float(line)
        elif "vllm:generation_tokens_total" in line :
            result["generation_tokens_total"] = _float(line)
        elif "vllm:tokens_generated_total" in line :
            if result["generation_tokens_total"] == 0.0:
                result["generation_tokens_total"] = _float(line)

        # Finished requests
        elif "vllm:request_success_total" in line and "{" not in line:
            result["requests_total"] += _float(line)

    result["hist_ttft"] = _parse_histogram(lines, "vllm:time_to_first_token_seconds")
    result["hist_itl"]  = _parse_histogram(lines, "vllm:time_per_output_token_seconds")
    result["hist_e2e"]  = _parse_histogram(lines, "vllm:e2e_request_latency_seconds")

    return result

async def fetch_metrics(session: aiohttp.ClientSession, base_url: str) -> Optional[dict]:
    try:
        async with session.get(
            f"{base_url}/metrics",
            timeout=aiohttp.ClientTimeout(total=5),
        ) as resp:
            if resp.status != 200:
                return None
            return scrape_raw(await resp.text())
    except Exception:
        return None

def compute_potential(
    current_gpu_pct: float,
    peak_gpu_pct: float,
    num_running: int,
    num_waiting: int,
    num_swapped: int,
) -> Optional[int]:

    total = num_running + num_waiting + num_swapped
    if peak_gpu_pct <= 0 or total == 0:
        return None                          # no baseline yet
    if current_gpu_pct >= 90.0:
        return 0                             # already saturated
    cost_per_req = current_gpu_pct / total      # anchored to session peak
    return int((90-current_gpu_pct)/cost_per_req)

def _lat_str(val: Optional[float], warn: float, crit: float) -> str:
    if val is None:
        return color(" n/a ", GREY)
    c = RED if val >= crit else YELLOW if val >= warn else GREEN
    return color(f"{val:7.3f}s", c)

def _tok_str(val: Optional[float]) -> str:
    if val is None:
        return color(" n/a", GREY)
    return color(f"{val:7.1f}", CYAN)

def render(
    metrics:      dict,
    prev:         Optional[dict],
    prev_ts:      Optional[float],
    base_url:     str,
    warn:         float,
    crit:         float,
    ts:           str,
    history:      deque,
    interval:     float,
    peak_gpu_pct: float = 0.0,
):
    gpu = metrics["gpu_cache_pct"]
    cpu = metrics["cpu_cache_pct"]

    prompt_tps = gen_tps = total_tps = req_ps = None
    dt_label = f"waiting for first delta…"
    if prev is not None and prev_ts is not None:
        dt = time.monotonic() - prev_ts
        if dt > 0:
            dt_label   = f"{dt:.1f}s window"
            prompt_tps = max(0.0, (metrics["prompt_tokens_total"]      - prev["prompt_tokens_total"])      / dt)
            gen_tps    = max(0.0, (metrics["generation_tokens_total"]  - prev["generation_tokens_total"])  / dt)
            req_ps     = max(0.0, (metrics["requests_total"]           - prev["requests_total"])           / dt)
            total_tps  = prompt_tps + gen_tps

    if gen_tps is not None:
        history.append(gen_tps)

    ttft_p50 = percentile_from_histogram(metrics["hist_ttft"], 50)
    ttft_p95 = percentile_from_histogram(metrics["hist_ttft"], 95)
    ttft_p99 = percentile_from_histogram(metrics["hist_ttft"], 99)
    itl_p50  = percentile_from_histogram(metrics["hist_itl"],  50)
    itl_p95  = percentile_from_histogram(metrics["hist_itl"],  95)
    e2e_p50  = percentile_from_histogram(metrics["hist_e2e"],  50)
    e2e_p95  = percentile_from_histogram(metrics["hist_e2e"],  95)

    print("\033[H\033[J", end="")

    print( f"{'vLLM Live Monitor'}")

    print(f" {ts} {base_url}")
    print()

    # KV cache
    print(f" {BOLD}KV Cache{RESET}")
    print(f" GPU {gpu:5.1f}% {kv_bar(gpu)}")
    if cpu > 0:
        print(f" CPU {cpu:5.1f}% {kv_bar(cpu)}")

    # Queue & Potential
    num_running = metrics["num_running"]
    num_waiting = metrics["num_waiting"]
    num_swapped = metrics["num_swapped"]
    total_reqs  = num_running + num_waiting + num_swapped

    potential = compute_potential(gpu, peak_gpu_pct, num_running, num_waiting, num_swapped)
    if potential is None:
        pot_str = color(" \u221e ", GREY) + color(" (no load baseline yet)", GREY)
    elif potential == 0:
        pot_str = color(" 0 ", RED)  + color(" (KV cache saturated \u226590%)", RED)
    else:
        pot_c   = RED if potential <= 2 else YELLOW if potential <= 10 else GREEN
        pot_str = color(f" {potential:>4d} ", pot_c)

    print()
    print(f" {BOLD}Queue & Capacity{RESET}")
    print(f" Running : {color(str(num_running).rjust(4), CYAN)} "
          f"Waiting : {color(str(num_waiting).rjust(4), YELLOW)} "
          f"Swapped : {color(str(num_swapped).rjust(4), GREY)}")
    print(f" Total active requests : {color(str(total_reqs).rjust(4), CYAN)}")
    print(f" Potential (until 90%) :{pot_str}{color(f' peak={peak_gpu_pct:.1f}%', GREY)}")

    # Throughput
    print()
    print(f" {BOLD}Throughput{RESET} {color(dt_label, GREY)}")
    print(f" Prompt tokens /s : {_tok_str(prompt_tps)}")
    print(f" Generation tok /s : {_tok_str(gen_tps)} ")
    print(f" Total tokens /s : {_tok_str(total_tps)}")
    if req_ps is not None:
        print(f" Requests /s : {color(f'{req_ps:7.2f}', CYAN)}")

    # Latency
    print()
    print(f" {BOLD}Latency{RESET} {color('cumulative since server start', GREY)}")
    if ttft_p50 is not None:
        print(f" TTFT Average={_lat_str(ttft_p50, 1.0, 5.0)} "
              f"Worst={_lat_str(ttft_p99, 2.0, 10.0)}")
    else:
        print(f" TTFT {'no data yet'}")

    if itl_p50 is not None:
        print(f" ITL Average={_lat_str(itl_p50, 0.05, 0.2)} "
              f"Worst={_lat_str(itl_p95, 0.05, 0.2)}")
    else:
        print(f" ITL {color('no data yet', GREY)}")

    if e2e_p50 is not None:
        print(f" E2E Average={_lat_str(e2e_p50, 5.0, 30.0)} "
              f"Worst={_lat_str(e2e_p95, 5.0, 30.0)}")
    else:
        print(f" E2E {color('no data yet', GREY)}")

    print()
    print(f"Ctrl-C to quit")

# ── KV cache summary ──────────────────────────────────────────────────────────

def print_kv_summary(gpu_history: list, cpu_history: list) -> None:
    print(color("\n\n KV Cache Session Summary\n", BOLD))

    if gpu_history:
        lo  = min(gpu_history)
        avg = sum(gpu_history) / len(gpu_history)
        hi  = max(gpu_history)

        def _pct(v: float) -> str:
            c = RED if v >= 90 else YELLOW if v >= 70 else GREEN
            return color(f"{v:5.1f}%", c)

        print(f" GPU KV Cache ({len(gpu_history)} samples)")
        print(f" Min : {_pct(lo)}")
        print(f" Avg : {_pct(avg)}")
        print(f" Max : {_pct(hi)}")
    else:
        print(f" GPU KV Cache {color('no data collected', GREY)}")

    if cpu_history:
        lo  = min(cpu_history)
        avg = sum(cpu_history) / len(cpu_history)
        hi  = max(cpu_history)

        def _pct(v: float) -> str:
            c = RED if v >= 90 else YELLOW if v >= 70 else GREEN
            return color(f"{v:5.1f}%", c)

        print()
        print(f" CPU KV Cache ({len(cpu_history)} samples)")
        print(f" Min : {_pct(lo)}")
        print(f" Avg : {_pct(avg)}")
        print(f" Max : {_pct(hi)}")

    print()

async def monitor(args: argparse.Namespace):
    base_url = f"http://{args.host}:{args.port}"
    print("\033[2J", end="")

    prev_metrics: Optional[dict] = None
    prev_ts:      Optional[float] = None
    history: deque = deque(maxlen=30)

    gpu_kv_history: list[float] = []
    cpu_kv_history: list[float] = []

    async with aiohttp.ClientSession() as session:
        while True:
            ts      = time.strftime("%H:%M:%S")
            metrics = await fetch_metrics(session, base_url)

            if metrics is None:
                print("\033[H\033[J", end="")
                print(color(f"\n [{ts}] Cannot reach {base_url}/metrics – retrying…", YELLOW))
            else:
                # Record KV cache sample before rendering
                gpu_kv_history.append(metrics["gpu_cache_pct"])
                if metrics["cpu_cache_pct"] > 0:
                    cpu_kv_history.append(metrics["cpu_cache_pct"])

                render(metrics, prev_metrics, prev_ts,
                       base_url, args.warn, args.crit, ts, history, args.interval)
                prev_metrics = metrics
                prev_ts      = time.monotonic()

            await asyncio.sleep(args.interval)

    return gpu_kv_history, cpu_kv_history

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live vLLM monitor – KV cache, queue, throughput, latency. No requests sent.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host",     default="0.0.0.0")
    p.add_argument("--port",     type=int,   default=8000)
    p.add_argument("--interval", type=float, default=5.0,
                   help="Poll interval in seconds (5s recommended for stable throughput deltas)")
    p.add_argument("--warn",     type=float, default=70.0, help="KV cache warn threshold (%%)")
    p.add_argument("--crit",     type=float, default=90.0, help="KV cache critical threshold (%%)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Thread-shared lists that the async loop populates; readable after cancel.
    gpu_kv_history: list[float] = []
    cpu_kv_history: list[float] = []

    async def _run():
        base_url = f"http://{args.host}:{args.port}"
        print("\033[2J", end="")

        prev_metrics: Optional[dict] = None
        prev_ts:      Optional[float] = None
        history: deque = deque(maxlen=30)

        async with aiohttp.ClientSession() as session:
            while True:
                ts      = time.strftime("%H:%M:%S")
                metrics = await fetch_metrics(session, base_url)

                if metrics is None:
                    print("\033[H\033[J", end="")
                    print(color(f"\n [{ts}] Cannot reach {base_url}/metrics – retrying…", YELLOW))
                else:
                    gpu_kv_history.append(metrics["gpu_cache_pct"])
                    if metrics["cpu_cache_pct"] > 0:
                        cpu_kv_history.append(metrics["cpu_cache_pct"])

                    render(metrics, prev_metrics, prev_ts,
                           base_url, args.warn, args.crit, ts, history, args.interval,
                           peak_gpu_pct=max(gpu_kv_history) if gpu_kv_history else 0.0)
                    prev_metrics = metrics
                    prev_ts      = time.monotonic()

                await asyncio.sleep(args.interval)

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print_kv_summary(gpu_kv_history, cpu_kv_history)
        print(color("Stopped.\n", YELLOW))
        sys.exit(0)