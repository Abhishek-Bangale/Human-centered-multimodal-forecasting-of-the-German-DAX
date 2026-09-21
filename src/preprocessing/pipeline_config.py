"""
Shared configuration for the raw-data pipeline (financial_data.py ->
download_gdelt_gkg.py -> process_gdelt_leads.py ->
prepare_headlines_for_sentiment.py -> preprocess.py).

Why this exists
----------------
The original scripts each hardcoded "the last 26 days, computed from
whatever `today` is when you run me" with no way to ask for a different
window, and each wrote/read files relative to the current working
directory using a handful of slightly-different ad hoc paths. That made
the pipeline impossible to point at a specific date range, and its
outputs impossible to find reliably.

This module gives every stage:
  * the SAME date range (via a shared CLI convention + a manifest file),
    so the price data and news data a run produces always cover the same
    period instead of silently drifting apart;
  * a single, repo-root-anchored raw-data directory, so every stage's
    output ends up somewhere predictable regardless of what directory
    you ran the script from;
  * a tiny JSON manifest recording what the current pipeline run actually
    used/produced, so a later stage (or preprocess.py) can pick up the
    exact files an earlier stage just built rather than guessing a name.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
GDELT_DIR = os.path.join(RAW_DIR, "gdelt_gkg_data")
LEADS_DIR = os.path.join(RAW_DIR, "processed_gdelt_leads")
MANIFEST_PATH = os.path.join(RAW_DIR, "pipeline_manifest.json")

DATE_FMT = "%Y-%m-%d"


def parse_date_range(default_days_back: int = 90, description: str = ""):
    """
    Shared CLI for every raw-pipeline stage that needs a date range.

    --end-date defaults to yesterday (GDELT/Yahoo Finance data for "today"
    isn't complete yet while today is still in progress).
    --days-back / --start-date are mutually exclusive; --days-back (the
    original scripts' behavior, just now a real, visible parameter instead
    of a hardcoded constant) is the default.

    Returns (start_date, end_date) as `date` objects.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--start-date", type=str, default=None,
                         help=f"First day to fetch, {DATE_FMT} (default: --end-date minus --days-back)")
    parser.add_argument("--end-date", type=str, default=None,
                         help=f"Last day to fetch, {DATE_FMT} (default: yesterday)")
    parser.add_argument("--days-back", type=int, default=default_days_back,
                         help=f"Used when --start-date is omitted (default: {default_days_back})")
    args, _ = parser.parse_known_args()

    end_date = (datetime.strptime(args.end_date, DATE_FMT).date()
                if args.end_date else (datetime.today() - timedelta(days=1)).date())
    start_date = (datetime.strptime(args.start_date, DATE_FMT).date()
                  if args.start_date else end_date - timedelta(days=args.days_back))

    if start_date > end_date:
        raise ValueError(f"--start-date ({start_date}) is after --end-date ({end_date})")

    return start_date, end_date


def resolve_date_range(default_days_back: int = 90, description: str = ""):
    """
    Like parse_date_range, but manifest-aware: if an earlier stage in this
    same pipeline run already wrote a date range to the manifest, reuse it
    exactly (rather than each script independently recomputing "today - N
    days" and risking a one-day drift, or a user re-running one stage
    later picking up a different window than the rest of the pipeline
    used). An explicit --start-date/--end-date/--days-back on THIS
    invocation always wins over the manifest.

    This is what keeps the price data (financial_data.py) and the news
    data (download_gdelt_gkg.py) aligned to the same window, which is the
    root cause documented in the README of the original pipeline
    producing a mismatched dataset.
    """
    explicit_override = any(
        flag in " ".join(sys.argv) for flag in ("--start-date", "--end-date", "--days-back")
    )
    if not explicit_override:
        manifest = load_manifest()
        if manifest.get("start_date") and manifest.get("end_date"):
            start_date = datetime.strptime(manifest["start_date"], DATE_FMT).date()
            end_date = datetime.strptime(manifest["end_date"], DATE_FMT).date()
            print(f"[pipeline_config] Reusing date range from {MANIFEST_PATH}: {start_date} to {end_date}")
            return start_date, end_date

    return parse_date_range(default_days_back=default_days_back, description=description)


def ensure_raw_dirs():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(GDELT_DIR, exist_ok=True)
    os.makedirs(LEADS_DIR, exist_ok=True)


def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def update_manifest(**kwargs):
    """Merge kwargs into the manifest and save it. Records e.g. the
    resolved date range and the path each stage wrote, so later stages
    (and a human debugging a run) can see exactly what a given run used
    without re-deriving it."""
    ensure_raw_dirs()
    manifest = load_manifest()
    manifest.update(kwargs)
    manifest["_last_updated_utc"] = datetime.utcnow().isoformat()
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest


class StageTimer:
    """Tiny context-manager/utility for printing real elapsed time per
    pipeline stage, so a run's actual wall-clock time is visible instead
    of being a black box (see README for why this matters — GDELT
    downloads and transformer sentiment scoring are the two genuinely
    slow, network/hardware-dependent steps, and there was previously no
    way to see where time was actually going)."""
    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        self._t0 = datetime.now()
        print(f"\n=== [{self._t0:%H:%M:%S}] START: {self.label} ===", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        elapsed = (datetime.now() - self._t0).total_seconds()
        status = "FAILED" if exc_type else "DONE"
        print(f"=== [{datetime.now():%H:%M:%S}] {status}: {self.label} ({elapsed:.1f}s) ===", flush=True)
        return False
