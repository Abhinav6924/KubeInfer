import time
from dataclasses import dataclass
from typing import Optional

class EMASlope:
    def __init__(self, value_half_life: float = 15.0, slope_half_life: float = 30.0):
        self.value_half_life = value_half_life
        self.slope_half_life = slope_half_life
        self.ema: Optional[float] = None
        self.slope: Optional[float] = None
        self._prev_ts: Optional[float] = None

    @staticmethod
    def _alpha(dt: float, half_life: float) -> float:
        if half_life <= 0:
            return 1.0
        return 1.0 - 0.5 ** (dt / half_life)

    def update(self, value: float, ts: Optional[float] = None) -> dict:
        ts = ts if ts is not None else time.monotonic()

        if self.ema is None:
            self.ema = float(value)
            self._prev_ts = ts
            return self.snapshot()

        dt = ts - self._prev_ts
        if dt <= 0:
            return self.snapshot()

        a_val = self._alpha(dt, self.value_half_life)
        new_ema = self.ema + a_val * (value - self.ema)

        raw_slope = (new_ema - self.ema) / dt
        a_slope = self._alpha(dt, self.slope_half_life)
        self.slope = raw_slope if self.slope is None else self.slope + a_slope * (raw_slope - self.slope)

        self.ema = new_ema
        self._prev_ts = ts
        return self.snapshot()

    def project(self, seconds_ahead: float) -> Optional[float]:
        if self.ema is None or self.slope is None:
            return None
        return self.ema + self.slope * seconds_ahead

    def time_to_threshold(self, threshold: float) -> Optional[float]:
        if self.ema is None or not self.slope:
            return None
        dt = (threshold - self.ema) / self.slope
        return dt if dt > 0 else None

    def snapshot(self) -> dict:
        return {"ema": self.ema, "slope_per_sec": self.slope}


@dataclass
class SaturationThresholds:
    warn_pct: float = 70.0
    crit_pct: float = 90.0
    saturation_pct: float = 100.0
    scale_now_lead_s: float = 60.0
    monitor_lead_s: float = 180.0

OK = "ok"
MONITOR = "monitor"
SCALE_NOW = "scale_now"
CRITICAL = "critical"


class KVSaturationPredictor:

    def __init__(
            self,
            thresholds: Optional[SaturationThresholds] = None,
            value_half_life: float = 15.0,
            slope_half_life: float = 30.0,
    ):
        self.thresholds = thresholds or SaturationThresholds()
        self._cache_es = EMASlope(value_half_life, slope_half_life)
        self._remaining_es = EMASlope(value_half_life, slope_half_life)

    def update(
            self,
            gpu_cache_pct: float,
            remaining_requests: Optional[float] = None,
            ts: Optional[float] = None,
    ) -> dict:
        ts = ts if ts is not None else time.monotonic()
        th = self.thresholds

        cache_snap = self._cache_es.update(gpu_cache_pct, ts)
        ema_pct, slope_pct = cache_snap["ema"], cache_snap["slope_per_sec"]

        eta_warn = self._eta_to_cache_threshold(ema_pct, th.warn_pct)
        eta_crit = self._eta_to_cache_threshold(ema_pct, th.crit_pct)
        eta_sat = self._eta_to_cache_threshold(ema_pct, th.saturation_pct)

        eta_requests_zero = None
        ema_remaining = None
        remaining_slope = None
        if remaining_requests is not None:
            rem_snap = self._remaining_es.update(remaining_requests, ts)
            ema_remaining, remaining_slope = rem_snap["ema"], rem_snap["slope_per_sec"]
            eta_requests_zero = 0.0 if ema_remaining <= 0 else self._remaining_es.time_to_threshold(0.0)

        candidates = [e for e in (eta_crit, eta_requests_zero) if e is not None]
        safe_serving_seconds = min(candidates) if candidates else None

        recommendation = self._recommend(ema_pct, safe_serving_seconds)

        return {
            "ema_cache_pct": ema_pct,
            "cache_slope_pct_per_sec": slope_pct,
            "eta_seconds_to_warn": eta_warn,
            "eta_seconds_to_crit": eta_crit,
            "eta_seconds_to_saturation": eta_sat,
            "ema_remaining_requests": ema_remaining,
            "remaining_requests_slope_per_sec": remaining_slope,
            "eta_seconds_to_zero_requests": eta_requests_zero,
            "safe_serving_seconds": safe_serving_seconds,
            "recommendation": recommendation,
        }

    def _eta_to_cache_threshold(self, ema: Optional[float], threshold: float) -> Optional[float]:
        if ema is None:
            return None
        if ema >= threshold:
            return 0.0
        return self._cache_es.time_to_threshold(threshold)

    def _recommend(self, ema_pct: Optional[float], safe_serving_seconds: Optional[float]) -> dict:
        th = self.thresholds

        if ema_pct is not None and ema_pct >= th.crit_pct:
            return {
                "level": CRITICAL,
                "message": f"KV cache already at {ema_pct:.1f}% (>= crit {th.crit_pct:.0f}%) ",
            }

        if safe_serving_seconds is not None:
            if safe_serving_seconds <= th.scale_now_lead_s:
                return {
                    "level": SCALE_NOW,
                    "message": f"Projected to hit saturation in ~{safe_serving_seconds:.0f}s "
                }
            if safe_serving_seconds <= th.monitor_lead_s:
                return {
                    "level": MONITOR,
                    "message": f"Projected to hit saturation in ~{safe_serving_seconds:.0f}s "
                }
            return {
                "level": OK,
                "message": f"~{safe_serving_seconds:.0f}s of runway before saturation at current trend — healthy.",
            }

        if ema_pct is not None and ema_pct >= th.warn_pct:
            return {
                "level": MONITOR,
                "message": f"KV cache at {ema_pct:.1f}% (>= warn {th.warn_pct:.0f}%) but not currently "
                           f"trending toward saturation — keep an eye on it.",
            }

        return {"level": OK, "message": "KV cache usage healthy, no scaling action needed."}


if __name__ == "__main__":
    #Test it to see if it works
    predictor = KVSaturationPredictor()
    predictor.thresholds.scale_now_lead_s = 30.0
    predictor.thresholds.monitor_lead_s = 90.0

    cache_series = [40, 45, 52, 58, 65, 71, 77, 83, 88, 92, 95, 90, 80, 65, 50]
    remaining_series = [30, 26, 21, 17, 12, 9, 6, 4, 2, 1, 0, 3, 9, 18, 28]

    t = 0.0
    for cache_pct, remaining in zip(cache_series, remaining_series):
        result = predictor.update(gpu_cache_pct=cache_pct, remaining_requests=remaining, ts=t)
        rec = result["recommendation"]
        safe = result["safe_serving_seconds"]
        safe_str = f"{safe:.0f}s" if safe is not None else "n/a"
        print(
            f"t={t:5.1f}  cache={cache_pct:5.1f}%  ema={result['ema_cache_pct']:5.1f}%  "
            f"safe_serving={safe_str:>6s}  {rec['level']:9s} {rec['message']}"
        )
        time.sleep(3)
        t += 5.0
