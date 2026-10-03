"""Simulated traffic for the churn API, to demonstrate monitoring and alerts (docs/open-questions.md, Q17).

The project has no real customers, so this script plays them: it samples customers from the Telco data
and sends them to the API at a chosen rate. Three modes:

- ``normal``: customers as they are in the data. Input PSI stays low and no drift alert should fire.
- ``drift``: a controlled shift, as if a new customer segment arrived: more month-to-month contracts,
  shorter tenure, higher monthly charges, more electronic-check payments. PSI rises above 0.2 for
  those inputs, and ``FeatureDrift`` / ``PredictionDrift`` fire after their ``for:`` windows.
- ``invalid``: a share of malformed requests (negative tenure, missing field, wrong type), which the
  API rejects with 422 and which drives ``HighValidationErrorRate``.

Standard library only, so it runs with the system ``python3`` on the host as well as in the trainer
container. Every request is labelled as simulated (``X-Request-ID: sim-...``) so the API logs show it.

Examples::

    python3 -m src.simulation.traffic --mode normal --rate 5 --duration 600
    python3 -m src.simulation.traffic --mode drift --rate 5 --duration 3600
    python3 -m src.simulation.traffic --mode drift --report      # offline PSI estimate, sends nothing
"""
from __future__ import annotations  # macOS system python3 is 3.9: keep `X | Y` hints unevaluated

import argparse
import csv
import json
import math
import random
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Dict

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "churn.csv"  # works from any cwd
DEFAULT_URL = "http://localhost:8088/api"
NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
MODES = ("normal", "drift", "invalid")
EPSILON = 1e-4  # same floor as the API's PSI, so empty buckets do not make PSI infinite

Customer = Dict[str, Any]
Sender = Callable[[str, Dict[str, Any], str], int]  # (path, JSON body, request id) -> HTTP status


def load_customers(path: Path = DATA_PATH) -> list[Customer]:
    """Customers of the processed Telco file, as the API expects them (raw columns, no target)."""
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    customers = []
    for row in rows:
        row.pop("Churn", None)
        row.pop("customerID", None)
        customer: Customer = dict(row)
        customer["SeniorCitizen"] = int(row["SeniorCitizen"])
        customer["tenure"] = int(row["tenure"])
        customer["MonthlyCharges"] = float(row["MonthlyCharges"])
        total = row.get("TotalCharges", "").strip()
        customer["TotalCharges"] = float(total) if total not in ("", "nan") else None
        customers.append(customer)
    return customers


def drift(customer: Customer, rng: random.Random, strength: float = 0.7) -> Customer:
    """Shift one customer towards a new segment; `strength` is the chance each change applies."""
    out = dict(customer)
    if rng.random() < strength:
        out["Contract"] = "Month-to-month"
    if rng.random() < strength:
        out["tenure"] = max(0, int(out["tenure"] * 0.25))
    if rng.random() < strength:
        out["MonthlyCharges"] = round(min(out["MonthlyCharges"] * 1.3 + 10, 150.0), 2)
    if rng.random() < strength:
        out["PaymentMethod"] = "Electronic check"
    # Keep the record consistent: total charges follow the (possibly shortened) tenure.
    out["TotalCharges"] = round(out["MonthlyCharges"] * out["tenure"], 2) if out["tenure"] > 0 else None
    return out


def corrupt(customer: Customer, rng: random.Random) -> dict[str, Any]:
    """A malformed version of `customer` that the API must reject with 422."""
    out: dict[str, Any] = dict(customer)
    kind = rng.choice(["negative_tenure", "missing_field", "wrong_type", "senior_out_of_range"])
    if kind == "negative_tenure":
        out["tenure"] = -1
    elif kind == "missing_field":
        out.pop("Contract")
    elif kind == "wrong_type":
        out["MonthlyCharges"] = "seventy"
    else:
        out["SeniorCitizen"] = 2
    return out


def generate(
    customers: list[Customer], mode: str, n: int, seed: int = 0, strength: float = 0.7, invalid_share: float = 0.3
) -> list[dict[str, Any]]:
    """`n` request bodies for `mode`, reproducible for a given seed."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    rng = random.Random(seed)
    bodies = []
    for _ in range(n):
        customer = rng.choice(customers)
        if mode == "drift":
            bodies.append(drift(customer, rng, strength))
        elif mode == "invalid" and rng.random() < invalid_share:
            bodies.append(corrupt(customer, rng))
        else:
            bodies.append(dict(customer))
    return bodies


def _bucketer(reference: list[Customer], feature: str, bins: int = 10) -> Callable[[Any], str]:
    """Bucket function for one feature: deciles of the reference for numbers, the value itself otherwise."""
    if feature in ("tenure", "MonthlyCharges", "TotalCharges"):
        values = sorted(float(c[feature]) for c in reference if c.get(feature) is not None)
        cuts = sorted({values[int(len(values) * i / bins)] for i in range(1, bins)})

        def bucket(value: Any) -> str:
            if value is None or isinstance(value, str):
                return "missing"
            return str(sum(float(value) >= cut for cut in cuts))

        return bucket
    return lambda value: str(value)


def psi(expected: Iterable[str], actual: Iterable[str]) -> float:
    """Population Stability Index between two samples of bucket labels."""
    exp, act = Counter(expected), Counter(actual)
    n_exp, n_act = sum(exp.values()), sum(act.values())
    total = 0.0
    for label in set(exp) | set(act):
        e = max(exp[label] / n_exp, EPSILON)
        a = max(act[label] / n_act, EPSILON)
        total += (a - e) * math.log(a / e)
    return total


def psi_report(reference: list[Customer], sample: list[dict[str, Any]]) -> dict[str, float]:
    """Offline estimate of the PSI the API will see, per feature, for a generated sample.

    The API buckets with the training split of the served model; this uses deciles of the whole
    file, so the numbers are close but not identical. Malformed bodies are left out (the API rejects them).
    """
    valid = [s for s in sample if all(f in s for f in reference[0]) and isinstance(s.get("tenure"), int)]
    report = {}
    for feature in reference[0]:
        bucket = _bucketer(reference, feature)
        report[feature] = round(psi((bucket(c[feature]) for c in reference), (bucket(s[feature]) for s in valid)), 4)
    return dict(sorted(report.items(), key=lambda item: -item[1]))


def http_sender(base_url: str, timeout: float = 10.0) -> Sender:
    """Sender that POSTs JSON to the API and returns the HTTP status (0 when the API cannot be reached)."""

    def send(path: str, body: Customer | dict[str, Any], request_id: str) -> int:
        request = urllib.request.Request(
            base_url.rstrip("/") + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "X-Request-ID": request_id},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.status
        except urllib.error.HTTPError as exc:
            return exc.code
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            return 0

    return send


def run(
    bodies: list[dict[str, Any]],
    send: Sender,
    rate: float,
    batch: int = 0,
    sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> Counter:
    """Send `bodies` at about `rate` requests per second; `batch` > 0 groups them for /v1/predict/batch.

    Returns how many requests ended with each status code.
    """
    statuses: Counter = Counter()
    groups = [bodies[i:i + batch] for i in range(0, len(bodies), batch)] if batch > 0 else [[b] for b in bodies]
    interval = 1.0 / rate if rate > 0 else 0.0
    for i, group in enumerate(groups, start=1):
        if batch > 0:
            status = send("/v1/predict/batch", {"customers": group}, f"sim-batch-{i:06d}")
        else:
            status = send("/v1/predict", group[0], f"sim-{i:06d}")
        statuses[status] += 1
        if status == 0 and statuses[0] == 1:
            log("API not reachable; is the port-forward to 8088 running (./run.sh)?")
        if i % 100 == 0:
            log(f"{i}/{len(groups)} sent: {dict(statuses)}")
        if interval:
            sleep(interval)
    return statuses


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send simulated customers to the churn API.")
    parser.add_argument("--mode", choices=MODES, default="normal")
    parser.add_argument("--url", default=DEFAULT_URL, help=f"API base URL (default {DEFAULT_URL})")
    parser.add_argument("--rate", type=float, default=5.0, help="requests per second (default 5)")
    parser.add_argument("--duration", type=float, default=600.0, help="seconds to run (default 600)")
    parser.add_argument("--count", type=int, help="number of customers; overrides --duration")
    parser.add_argument("--batch", type=int, default=0, help="send groups of this size to /v1/predict/batch")
    parser.add_argument("--strength", type=float, default=0.7, help="drift mode: chance each shift applies")
    parser.add_argument("--invalid-share", type=float, default=0.3, help="invalid mode: share of bad requests")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=DATA_PATH)
    parser.add_argument("--report", action="store_true", help="print the expected PSI per feature and exit")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None, send: Sender | None = None) -> int:
    args = parse_args(argv)
    customers = load_customers(args.data)
    count = args.count or max(1, int(args.rate * args.duration))
    bodies = generate(customers, args.mode, count, args.seed, args.strength, args.invalid_share)
    if args.report:
        print(f"Expected input PSI for {count} '{args.mode}' customers (>0.2 fires FeatureDrift after 30 min):")
        for feature, value in psi_report(customers, bodies).items():
            print(f"  {feature:18s} {value:.3f}{'  <- drift' if value > 0.2 else ''}")
        return 0
    print(f"Sending {count} '{args.mode}' customers to {args.url} at {args.rate}/s (simulated traffic).")
    statuses = run(bodies, send or http_sender(args.url), args.rate, args.batch)
    print(f"Done: {dict(statuses)}")
    return 1 if statuses.get(0) == sum(statuses.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
