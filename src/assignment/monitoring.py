"""
Assignment 11 — Monitoring & Alerts starter (TODO).

Tracks block rate, rate-limit hits, judge fail rate.
Fires alerts when thresholds are exceeded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def default_metrics_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "metrics.json")


@dataclass
class Alert:
    metric: str
    value: float
    threshold: float
    message: str


@dataclass
class MonitoringAlert:
    """Aggregate counters from pipeline plugins and emit alerts."""

    block_rate_threshold: float = 0.5
    rate_limit_hit_threshold: int = 5
    judge_fail_rate_threshold: float = 0.3
    alerts: list[Alert] = field(default_factory=list)

    # Counters — update these from your pipeline after each request
    total_requests: int = 0
    blocked_requests: int = 0
    rate_limit_hits: int = 0
    judge_checks: int = 0
    judge_fails: int = 0

    def check_metrics(self) -> list[Alert]:
        """Compute rates; append (and return) new Alerts when thresholds are exceeded."""
        snap = self.snapshot()
        candidates = [
            (
                "block_rate",
                snap["block_rate"],
                self.block_rate_threshold,
                snap["block_rate"] > self.block_rate_threshold,
                "Block rate above threshold — possible attack campaign or "
                "over-aggressive filters (check false positives).",
            ),
            (
                "rate_limit_hits",
                float(self.rate_limit_hits),
                float(self.rate_limit_hit_threshold),
                self.rate_limit_hits >= self.rate_limit_hit_threshold,
                "Many rate-limit hits — possible flooding / cost attack.",
            ),
            (
                "judge_fail_rate",
                snap["judge_fail_rate"],
                self.judge_fail_rate_threshold,
                snap["judge_fail_rate"] > self.judge_fail_rate_threshold,
                "LLM judge rejects many responses — model may be leaking or "
                "hallucinating.",
            ),
        ]
        fired = {a.metric for a in self.alerts}
        new_alerts = []
        for metric, value, threshold, exceeded, message in candidates:
            if exceeded and metric not in fired:
                alert = Alert(
                    metric=metric,
                    value=round(value, 4),
                    threshold=threshold,
                    message=message,
                )
                self.alerts.append(alert)
                new_alerts.append(alert)
                print(f"[ALERT] {metric}={value:.2f} (threshold {threshold}) — {message}")
        return new_alerts

    def record(self, *, blocked: bool, rate_limited: bool = False,
               judge_checked: bool = False, judge_failed: bool = False):
        """Update counters after one request."""
        self.total_requests += 1
        if blocked:
            self.blocked_requests += 1
        if rate_limited:
            self.rate_limit_hits += 1
        if judge_checked:
            self.judge_checks += 1
            if judge_failed:
                self.judge_fails += 1

    def export_json(self, filepath: str | None = None):
        """Write metrics + alerts to JSON under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_metrics_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        data = self.snapshot()
        data["thresholds"] = {
            "block_rate": self.block_rate_threshold,
            "rate_limit_hits": self.rate_limit_hit_threshold,
            "judge_fail_rate": self.judge_fail_rate_threshold,
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(path)

    def snapshot(self) -> dict:
        block_rate = (
            self.blocked_requests / self.total_requests
            if self.total_requests
            else 0.0
        )
        judge_fail_rate = (
            self.judge_fails / self.judge_checks if self.judge_checks else 0.0
        )
        return {
            "total_requests": self.total_requests,
            "blocked_requests": self.blocked_requests,
            "block_rate": block_rate,
            "rate_limit_hits": self.rate_limit_hits,
            "judge_checks": self.judge_checks,
            "judge_fails": self.judge_fails,
            "judge_fail_rate": judge_fail_rate,
            "alerts": [
                {
                    "metric": a.metric,
                    "value": a.value,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
        }
