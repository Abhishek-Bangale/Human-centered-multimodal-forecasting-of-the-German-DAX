"""
get_input_for_prediction.py — orchestrates the full raw-data pipeline.

Runs, in order: financial_data.py -> download_gdelt_gkg.py ->
process_gdelt_leads.py -> prepare_headlines_for_sentiment.py ->
preprocess.py.

Any arguments you pass to THIS script are forwarded to every stage that
understands them (--start-date, --end-date, --days-back — see
pipeline_config.py), so:

    python get_input_for_prediction.py --days-back 60

fetches/processes the last 60 days end-to-end, instead of being stuck on
the previous hardcoded "last 26 days". Can be run from any directory —
every stage anchors its paths to the repo root, not to the current
working directory.

Each stage's real wall-clock time is printed as it runs (via
pipeline_config.StageTimer inside each script, plus the summary this
script prints at the end) so a run's actual duration is visible instead
of being a black box — see the README for why that matters (GDELT
downloads and transformer sentiment scoring are genuinely slow and
network/hardware-dependent; there's no way to make that instant, only to
make it visible and non-redundant).
"""
import os
import subprocess
import sys
import time

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EXTRA_ARGS = sys.argv[1:]  # forwarded to every stage, e.g. --days-back 60

STAGES = [
    ("financial_data.py", "Stage 1/5: fetch DAX price data"),
    ("download_gdelt_gkg.py", "Stage 2/5: download GDELT GKG dumps"),
    ("process_gdelt_leads.py", "Stage 3/5: extract DAX-relevant headline leads"),
    ("prepare_headlines_for_sentiment.py", "Stage 4/5: clean/dedupe headlines"),
    ("preprocess.py", "Stage 5/5: sentiment scoring + final dataset assembly"),
]


def run_stage(script_name: str, label: str) -> float:
    script_path = os.path.join(THIS_DIR, script_name)
    cmd = [sys.executable, script_path] + EXTRA_ARGS
    print(f"\n{'='*80}\n{label}  ({script_name})\n{'='*80}", flush=True)

    t0 = time.time()
    result = subprocess.run(cmd)
    elapsed = time.time() - t0

    print(f"--- {label} finished in {elapsed:.1f}s (exit code {result.returncode}) ---", flush=True)
    if result.returncode != 0:
        print(f"\n❌ {script_name} failed (exit code {result.returncode}). Stopping pipeline here.")
        sys.exit(result.returncode)
    return elapsed


def main():
    pipeline_t0 = time.time()
    timings = []

    for script_name, label in STAGES:
        elapsed = run_stage(script_name, label)
        timings.append((label, elapsed))

    total_elapsed = time.time() - pipeline_t0

    print(f"\n{'='*80}\nPIPELINE COMPLETE — timing summary\n{'='*80}")
    for label, elapsed in timings:
        print(f"  {elapsed:7.1f}s  {label}")
    print(f"  {total_elapsed:7.1f}s  TOTAL")
    print("\nCheck the repo's data/ folder for the output parquet files "
          "(dax_final_with_sentiment_*.parquet) and data/raw/pipeline_manifest.json "
          "for exactly what date range and input files this run used.")


if __name__ == "__main__":
    main()
