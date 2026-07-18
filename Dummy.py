#!/usr/bin/env python3
"""
Live vLLM monitor - KV cache, queue, throughput, latency.

Auto-detects --num-layers/--num-kv-heads/--head-dim (from the model's
config.json via transformers) and --total-kv-tokens/--block-size (from
vLLM's /metrics cache_config_info gauge) on startup, so it can usually
just be run as:

    python vllm_monitor.py --host 0.0.0.0 --port 8000

Manual flags still override auto-detection if given. Use --no-autodetect
to disable it entirely (e.g. old vLLM builds without cache_config_info).
"""

import argparse
import asyncio
import math
import re
import sys
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

import aiohttp
import requests

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
GREY = "\033[90m"


def color(text: str, c: str) -> str:
    return f"{c}{text}{RESET}"


def kv_bar(pct: float, width: int = 36) -> str:
    filled = int(pct / 100 * width)
    c = RED if pct >= 90 else YELLOW if pct >= 70 else GREEN
    return color("█" * filled, c) + color("░" * (width - filled), GREY)


def _parse_histogram(lines: list, metric_name: str) -> dict:
    buckets: dict = {}
    total_sum = 0.0
    total_count = 0.0
    for line in lines:
        if line.startswith("#") or metric_name not in line:
            continue
        if f"{metric_name}_bucket" in line:
            try:
                le = float(line.split('le="')[1].split('"')[0])
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
    total = hist.get("_count", 0)
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
        "gpu_cache_pct": 0.0,
        "cpu_cache_pct": 0.0,
        "num_running": 0,
        "num_waiting": 0,
        "num_swapped": 0,
        "prompt_tokens_total": 0.0,
        "generation_tokens_total": 0.0,
        "requests_total": 0.0,
        "hist_ttft": {},
        "hist_itl": {},
        "hist_e2e": {},
        # Live KV capacity, if this vLLM version exposes it directly.
        # Different vLLM versions/forks have used different metric names for
        # this (it's a recent, still-evolving addition upstream), so we watch
        # for all known variants and let the caller pick whichever is present.
        "live_total_blocks": None,
        "live_total_tokens": None,
        "live_free_tokens": None,
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
        elif "vllm:prompt_tokens_total" in line:
            result["prompt_tokens_total"] = _float(line)
        elif "vllm:generation_tokens_total" in line:
            result["generation_tokens_total"] = _float(line)
        elif "vllm:tokens_generated_total" in line:
            if result["generation_tokens_total"] == 0.0:
                result["generation_tokens_total"] = _float(line)

        # Finished requests – sum across all finish_reason label variants.
        elif "vllm:request_success_total" in line:
            result["requests_total"] += _float(line)

        # Live KV capacity metrics – only present on newer vLLM builds.
        elif "vllm:num_kv_cache_total_blocks" in line:
            result["live_total_blocks"] = _float(line)
        elif "vllm:kv_cache_total_tokens" in line:
            result["live_total_tokens"] = _float(line)
        elif "vllm:kv_cache_free_tokens" in line:
            result["live_free_tokens"] = _float(line)

    result["hist_ttft"] = _parse_histogram(lines, "vllm:time_to_first_token_seconds")
    result["hist_itl"] = _parse_histogram(lines, "vllm:time_per_output_token_seconds")
    result["hist_e2e"] = _parse_histogram(lines, "vllm:e2e_request_latency_seconds")

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


# ---------------------------------------------------------------------------
# Auto-detection (merged from vllm_autodetect.py)
# ---------------------------------------------------------------------------

def get_served_model_id(base_url: str) -> Optional[str]:
    """Ask vLLM's OpenAI-compatible endpoint what model it's serving."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        resp.raise_for_status()
        data = resp.json()
        models = data.get("data", [])
        if models:
            return models[0]["id"]
    except Exception as e:
        print(f"  ! couldn't reach {base_url}/v1/models ({e})", file=sys.stderr)
    return None


def resolve_architecture(model_id: str) -> Optional[dict]:
    """Pull num_layers / num_kv_heads / head_dim from the model's config.json.

    Handles VL/multimodal configs where the language-model params are
    nested under `text_config` rather than top-level (Qwen3-VL, and most
    other recent VLMs, do this).
    """
    try:
        from transformers import AutoConfig
    except ImportError:
        print("  ! `transformers` not installed. Run: pip install transformers --break-system-packages",
              file=sys.stderr)
        return None

    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    except Exception as e:
        print(f"  ! couldn't load config for '{model_id}' ({e})", file=sys.stderr)
        return None

    text_cfg = getattr(cfg, "text_config", None) or cfg

    try:
        num_layers = text_cfg.num_hidden_layers
        num_attention_heads = text_cfg.num_attention_heads
        num_kv_heads = getattr(text_cfg, "num_key_value_heads", num_attention_heads)
        head_dim = getattr(text_cfg, "head_dim", None)
        if head_dim is None:
            head_dim = text_cfg.hidden_size // num_attention_heads
    except AttributeError as e:
        print(f"  ! config for '{model_id}' is missing an expected field ({e})", file=sys.stderr)
        return None

    return {
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
    }


def parse_cache_config_info(metrics_text: str) -> Optional[dict]:
    """Parse the vllm:cache_config_info{...} label set, if present."""
    for line in metrics_text.splitlines():
        if line.startswith("#") or "vllm:cache_config_info" not in line:
            continue
        m = re.search(r"\{(.*)\}", line)
        if not m:
            continue
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        return labels
    return None


def resolve_capacity(base_url: str) -> dict:
    """Try to get block_size + total_kv_tokens live from /metrics."""
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=5)
        resp.raise_for_status()
    except Exception as e:
        return {"ok": False, "reason": f"couldn't reach {base_url}/metrics ({e})"}

    labels = parse_cache_config_info(resp.text)
    if labels is None:
        return {"ok": False,
                "reason": "no vllm:cache_config_info metric found — vLLM version may be too old. "
                          "Fall back to reading '# GPU blocks' / 'GPU KV cache size' from the "
                          "server's startup log instead."}

    block_size = labels.get("block_size")
    num_gpu_blocks = labels.get("num_gpu_blocks")

    if not block_size:
        return {"ok": False, "reason": "cache_config_info present but missing block_size label"}

    if not num_gpu_blocks or num_gpu_blocks == "None":
        return {"ok": False,
                "reason": "cache_config_info present but num_gpu_blocks is 'None' — a known gap in "
                          "some vLLM V1 releases. Read 'GPU KV cache size: N tokens' from the "
                          "server's startup log instead and pass it via --total-kv-tokens."}

    block_size = int(block_size)
    num_gpu_blocks = int(num_gpu_blocks)
    return {
        "ok": True,
        "block_size": block_size,
        "num_gpu_blocks": num_gpu_blocks,
        "total_kv_tokens": num_gpu_blocks * block_size,
    }


def autodetect(args: argparse.Namespace) -> None:
    """Fill in missing --num-layers/--num-kv-heads/--head-dim/--total-kv-tokens/
    --block-size on args in place, by querying the running vLLM server.
    Only fills fields the user didn't already set manually.
    """
    base_url = f"http://{args.host}:{args.port}"
    need_arch = not (args.num_layers and args.num_kv_heads and args.head_dim)
    need_cap = not (args.total_kv_tokens or args.num_gpu_blocks)

    if not (need_arch or need_cap):
        return

    print("Auto-detecting KV-capacity flags...")

    if need_arch:
        model_id = args.model or get_served_model_id(base_url)
        if model_id:
            print(f"  server reports model: {model_id}")
            arch = resolve_architecture(model_id)
            if arch:
                args.num_layers = args.num_layers or arch["num_layers"]
                args.num_kv_heads = args.num_kv_heads or arch["num_kv_heads"]
                args.head_dim = args.head_dim or arch["head_dim"]
                print(f"  num_layers={args.num_layers}  num_kv_heads={args.num_kv_heads}  "
                      f"head_dim={args.head_dim}")
            else:
                print("  ! could not resolve architecture — pass --num-layers/--num-kv-heads/"
                      "--head-dim manually, or --model to override the served id")
        else:
            print("  ! could not determine served model id — pass --model explicitly")

    if need_cap:
        cap = resolve_capacity(base_url)
        if cap["ok"]:
            args.total_kv_tokens = cap["total_kv_tokens"]
            args.block_size = cap["block_size"]
            print(f"  block_size={cap['block_size']}  num_gpu_blocks={cap['num_gpu_blocks']}  "
                  f"total_kv_tokens={cap['total_kv_tokens']}")
        else:
            print(f"  ! {cap['reason']}")
            print("  (forecast section will note capacity is unavailable; live /metrics "
                  "kv_cache_total_tokens, if your vLLM exposes it, is still used every poll "
                  "regardless of this.)")

    print()


# ---------------------------------------------------------------------------
# KV capacity forecasting
# ---------------------------------------------------------------------------

@dataclass
class KVCapacityConfig:
    block_size: int  # tokens per KV block (vLLM default: 16)
    bytes_per_token: float  # derived from model architecture
    total_kv_tokens: Optional[int]  # None if unknown -> forecast disabled
    source: str = "unset"  # "log" | "estimated" | "unset"

    @property
    def total_blocks(self) -> Optional[int]:
        if self.total_kv_tokens is None:
            return None
        return int(self.total_kv_tokens // self.block_size)


def kv_bytes_per_token(
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype_bytes: int = 2,  # 2 = fp16/bf16, 1 = fp8
) -> float:
    """K + V, per layer, per token."""
    return 2 * num_layers * num_kv_heads * head_dim * dtype_bytes


def estimate_total_kv_tokens_from_memory(
        gpu_mem_gb: float,
        gpu_mem_utilization: float,
        model_params_b: float,
        weight_dtype_bytes: int,
        bytes_per_token: float,
        activation_overhead_gb: float = 2.0,
) -> int:
    total_bytes = gpu_mem_gb * (1024 ** 3) * gpu_mem_utilization
    weight_bytes = model_params_b * 1e9 * weight_dtype_bytes
    kv_bytes = max(0.0, total_bytes - weight_bytes - activation_overhead_gb * (1024 ** 3))
    return int(kv_bytes // bytes_per_token)


def resolve_total_blocks(
        metrics: dict,
        kv_config: Optional[KVCapacityConfig],
        block_size: int,
) -> tuple:
    """Figure out total KV blocks, preferring live data straight from the
    /metrics endpoint over anything the user configured on the CLI.

    Priority:
      1. vllm:kv_cache_total_tokens (live, exact, no flags needed)
      2. vllm:num_kv_cache_total_blocks (live, exact, no flags needed)
      3. kv_config.total_blocks (from --total-kv-tokens / --num-gpu-blocks /
         estimated from GPU memory / auto-detected)
      4. None -> forecast section explains why it's unavailable
    """
    live_tokens = metrics.get("live_total_tokens")
    if live_tokens:
        return int(live_tokens // block_size), "live from /metrics (kv_cache_total_tokens)"

    live_blocks = metrics.get("live_total_blocks")
    if live_blocks:
        return int(live_blocks), "live from /metrics (num_kv_cache_total_blocks)"

    if kv_config is not None and kv_config.total_blocks is not None:
        return kv_config.total_blocks, kv_config.source

    return None, "no KV capacity available (live metric absent, and no --total-kv-tokens/--num-gpu-blocks/--gpu-mem-gb given)"


def compute_capacity_forecast(
        gpu_cache_pct: float,
        total_blocks: Optional[int],
        capacity_source: str,
        block_size: int,
        avg_prompt_len: Optional[float],
        avg_output_len: Optional[float],
) -> dict:
    if total_blocks is None:
        return {"remaining_requests": None, "reason": capacity_source}
    if not avg_prompt_len or not avg_output_len:
        return {"remaining_requests": None, "reason": "waiting for enough finished requests to profile avg size"}

    used_blocks = round(gpu_cache_pct / 100.0 * total_blocks)
    free_blocks = max(0, total_blocks - used_blocks)

    avg_total_len = avg_prompt_len + avg_output_len
    blocks_per_request = max(1, math.ceil(avg_total_len / block_size))
    remaining_requests = free_blocks // blocks_per_request

    return {
        "remaining_requests": remaining_requests,
        "free_blocks": free_blocks,
        "total_blocks": total_blocks,
        "blocks_per_request": blocks_per_request,
        "avg_prompt_len": avg_prompt_len,
        "avg_output_len": avg_output_len,
        "source": capacity_source,
    }


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
        metrics: dict,
        prev: Optional[dict],
        prev_ts: Optional[float],
        base_url: str,
        warn: float,
        crit: float,
        ts: str,
        history: deque,
        interval: float,
        kv_config: Optional[KVCapacityConfig] = None,
        block_size: int = 16,
):
    gpu = metrics["gpu_cache_pct"]
    cpu = metrics["cpu_cache_pct"]

    prompt_tps = gen_tps = total_tps = req_ps = None
    dt_label = f"waiting for first delta…"
    if prev is not None and prev_ts is not None:
        dt = time.monotonic() - prev_ts
        if dt > 0:
            dt_label = f"{dt:.1f}s window"
            prompt_tps = max(0.0, (metrics["prompt_tokens_total"] - prev["prompt_tokens_total"]) / dt)
            gen_tps = max(0.0, (metrics["generation_tokens_total"] - prev["generation_tokens_total"]) / dt)
            req_ps = max(0.0, (metrics["requests_total"] - prev["requests_total"]) / dt)
            total_tps = prompt_tps + gen_tps

    if gen_tps is not None:
        history.append(gen_tps)

    ttft_p50 = percentile_from_histogram(metrics["hist_ttft"], 50)
    ttft_p95 = percentile_from_histogram(metrics["hist_ttft"], 95)
    ttft_p99 = percentile_from_histogram(metrics["hist_ttft"], 99)
    itl_p50 = percentile_from_histogram(metrics["hist_itl"], 50)
    itl_p95 = percentile_from_histogram(metrics["hist_itl"], 95)
    e2e_p50 = percentile_from_histogram(metrics["hist_e2e"], 50)
    e2e_p95 = percentile_from_histogram(metrics["hist_e2e"], 95)

    print("\033[H\033[J", end="")

    print(f"{'vLLM Live Monitor'}")

    print(f" {ts} {base_url}")
    print()

    # KV cache
    print(f" {BOLD}KV Cache{RESET}")
    print(f" GPU {gpu:5.1f}% {kv_bar(gpu)}")
    if cpu > 0:
        print(f" CPU {cpu:5.1f}% {kv_bar(cpu)}")

    # Queue
    num_running = metrics["num_running"]
    num_waiting = metrics["num_waiting"]
    num_swapped = metrics["num_swapped"]
    total_reqs = num_running + num_waiting + num_swapped

    print()
    print(f" {BOLD}Queue{RESET}")
    print(f" Running : {color(str(num_running).rjust(4), CYAN)} "
          f"Waiting : {color(str(num_waiting).rjust(4), YELLOW)} "
          f"Swapped : {color(str(num_swapped).rjust(4), GREY)}")
    print(f" Total active requests : {color(str(total_reqs).rjust(4), CYAN)}")

    # Capacity forecast – block-accounting model.
    total_blocks, capacity_source = resolve_total_blocks(metrics, kv_config, block_size)

    avg_prompt_len = (metrics["prompt_tokens_total"] / metrics["requests_total"]
                      if metrics["requests_total"] else None)
    avg_output_len = (metrics["generation_tokens_total"] / metrics["requests_total"]
                      if metrics["requests_total"] else None)
    forecast = compute_capacity_forecast(gpu, total_blocks, capacity_source, block_size,
                                          avg_prompt_len, avg_output_len)

    print()
    print(f" {BOLD}KV Capacity Forecast{RESET}")
    if forecast["remaining_requests"] is None:
        print(f" {color(forecast['reason'], GREY)}")
    else:
        n = forecast["remaining_requests"]
        c = RED if n <= 2 else YELLOW if n <= 10 else GREEN
        avg_p = forecast["avg_prompt_len"]
        avg_o = forecast["avg_output_len"]
        bpr = forecast["blocks_per_request"]
        detail = f"(avg {avg_p:.0f}+{avg_o:.0f} tok, {bpr} blk/req)"
        print(f" Room for ~{color(str(n), c)} more requests like the ones seen so far "
              f"{color(detail, GREY)}")
        source_label = f"({forecast['source']})"
        print(f" Free blocks : {color(str(forecast['free_blocks']), CYAN)} / {forecast['total_blocks']} "
              f"{color(source_label, GREY)}")

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


def print_kv_summary(gpu_history: list, cpu_history: list) -> None:
    print(color("\n\n KV Cache Session Summary\n", BOLD))

    if gpu_history:
        lo = min(gpu_history)
        avg = sum(gpu_history) / len(gpu_history)
        hi = max(gpu_history)

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
        lo = min(cpu_history)
        avg = sum(cpu_history) / len(cpu_history)
        hi = max(cpu_history)

        def _pct(v: float) -> str:
            c = RED if v >= 90 else YELLOW if v >= 70 else GREEN
            return color(f"{v:5.1f}%", c)

        print()
        print(f" CPU KV Cache ({len(cpu_history)} samples)")
        print(f" Min : {_pct(lo)}")
        print(f" Avg : {_pct(avg)}")
        print(f" Max : {_pct(hi)}")

    print()


def build_kv_config(args: argparse.Namespace) -> Optional[KVCapacityConfig]:
    if not (args.num_layers and args.num_kv_heads and args.head_dim):
        return None  # no architecture available -> forecast section will say so

    bpt = kv_bytes_per_token(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        dtype_bytes=args.kv_dtype_bytes,
    )

    if args.total_kv_tokens:
        return KVCapacityConfig(
            block_size=args.block_size,
            bytes_per_token=bpt,
            total_kv_tokens=args.total_kv_tokens,
            source="from --total-kv-tokens (log/auto-detected)",
        )

    if args.num_gpu_blocks:
        return KVCapacityConfig(
            block_size=args.block_size,
            bytes_per_token=bpt,
            total_kv_tokens=args.num_gpu_blocks * args.block_size,
            source="from --num-gpu-blocks (log)",
        )

    if args.gpu_mem_gb and args.model_params_b:
        total_tokens = estimate_total_kv_tokens_from_memory(
            gpu_mem_gb=args.gpu_mem_gb,
            gpu_mem_utilization=args.gpu_mem_utilization,
            model_params_b=args.model_params_b,
            weight_dtype_bytes=args.weight_dtype_bytes,
            bytes_per_token=bpt,
            activation_overhead_gb=args.activation_overhead_gb,
        )
        return KVCapacityConfig(
            block_size=args.block_size,
            bytes_per_token=bpt,
            total_kv_tokens=total_tokens,
            source="estimated from GPU mem (approximate)",
        )

    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live vLLM monitor – KV cache, queue, throughput, latency. No requests sent. "
                     "Auto-detects architecture/capacity flags from the running server unless "
                     "--no-autodetect is given.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--interval", type=float, default=5.0,
                   help="Poll interval in seconds (5s recommended for stable throughput deltas)")
    p.add_argument("--warn", type=float, default=70.0, help="KV cache warn threshold (%%)")
    p.add_argument("--crit", type=float, default=90.0, help="KV cache critical threshold (%%)")
    p.add_argument("--no-autodetect", action="store_true",
                   help="Skip auto-detection of architecture/capacity flags on startup")
    p.add_argument("--model", default=None,
                   help="Override the model id/path used for architecture auto-detection "
                        "(default: ask the server via /v1/models)")

    g = p.add_argument_group("KV capacity forecast")
    g.add_argument("--block-size", type=int, default=16,
                   help="vLLM KV block size in tokens (--block-size at vLLM startup). "
                        "Auto-detected from /metrics if not given.")
    g.add_argument("--num-layers", type=int, default=None, help="Model num hidden layers (auto-detected)")
    g.add_argument("--num-kv-heads", type=int, default=None, help="Model num KV heads, GQA/MQA-aware (auto-detected)")
    g.add_argument("--head-dim", type=int, default=None, help="Model attention head dim (auto-detected)")
    g.add_argument("--kv-dtype-bytes", type=int, default=2, help="KV cache dtype size: 2=fp16/bf16, 1=fp8")

    g.add_argument("--total-kv-tokens", type=int, default=None,
                   help="Exact total KV token capacity. Auto-detected from /metrics "
                        "cache_config_info if not given.")
    g.add_argument("--num-gpu-blocks", type=int, default=None,
                   help="Exact '# GPU blocks: N' value straight from the vLLM startup log. "
                        "Same fallback priority as --total-kv-tokens.")

    # fallback: approximate from raw GPU memory
    g.add_argument("--gpu-mem-gb", type=float, default=None, help="Total GPU memory in GB (fallback)")
    g.add_argument("--gpu-mem-utilization", type=float, default=0.9,
                   help="vLLM --gpu-memory-utilization value (fallback)")
    g.add_argument("--model-params-b", type=float, default=None, help="Model size in billions of params (fallback)")
    g.add_argument("--weight-dtype-bytes", type=int, default=2, help="Model weight dtype size (fallback)")
    g.add_argument("--activation-overhead-gb", type=float, default=2.0,
                   help="Rough activation/CUDA-graph memory reserve to subtract (fallback)")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.no_autodetect:
        autodetect(args)

    kv_config = build_kv_config(args)

    gpu_kv_history: list[float] = []
    cpu_kv_history: list[float] = []


    async def _run():
        base_url = f"http://{args.host}:{args.port}"
        print("\033[2J", end="")

        prev_metrics: Optional[dict] = None
        prev_ts: Optional[float] = None
        history: deque = deque(maxlen=30)

        async with aiohttp.ClientSession() as session:
            while True:
                ts = time.strftime("%H:%M:%S")
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
                           kv_config=kv_config, block_size=args.block_size)
                    prev_metrics = metrics
                    prev_ts = time.monotonic()

                await asyncio.sleep(args.interval)


    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        print_kv_summary(gpu_kv_history, cpu_kv_history)
        print(color("Stopped.\n", YELLOW))
        sys.exit(0)
