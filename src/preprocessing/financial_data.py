"""
financial_data.py — Stage 1 of the raw-data pipeline.

Fetches DAX (^GDAXI) index OHLCV data for a date range and saves it to
data/raw/ (repo-root-anchored, not CWD-relative). The date range defaults
to the last 26 days but is a real parameter now — pass --days-back N or
--start-date/--end-date to fetch a different window; see pipeline_config.py.

The output filename and date range are recorded in the shared pipeline
manifest (data/raw/pipeline_manifest.json) so preprocess.py (stage 5) can
find and use this exact file instead of always falling back to the fixed
historical dataset shipped in data/.
"""

import os
import sys
import yfinance as yf
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import resolve_date_range, ensure_raw_dirs, update_manifest, RAW_DIR


def fetch_dax_index_data(start_date: str, end_date: str, ticker: str = "^GDAXI") -> pd.DataFrame:
    print(f"Fetching data for ticker: {ticker} from {start_date} to {end_date}")
    df = yf.download(
        tickers=ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=True
    )
    if not df.empty:
        df.index.name = "Date"
    return df


def main():
    dax_index_ticker = "^GDAXI"
    ensure_raw_dirs()

    start_date, end_date = resolve_date_range(
        # 90 calendar days, not the original 26: the model's feature
        # engineering needs ~13 trading days of rolling-window warm-up
        # (RSI-14 is the binding constraint) before its 30-day sequence
        # window even starts, so the practical minimum is ~44 trading days
        # (~62-65 calendar days accounting for weekends/holidays) to
        # produce even ONE predictable sequence. 90 gives a safety margin
        # — see README "Updating to the latest data".
        default_days_back=90,
        description="Fetch DAX OHLCV price data for a date range."
    )
    # yfinance's `end` is exclusive, so add a day to actually include end_date.
    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()
    yf_end_str = (end_date + pd.Timedelta(days=1)).date().isoformat()

    print(f"Attempting to fetch DAX index data for ticker {dax_index_ticker} from {start_date_str} to {end_date_str}.")

    dax_df = fetch_dax_index_data(start_date_str, yf_end_str, ticker=dax_index_ticker)

    if dax_df.empty:
        print(f"No data returned for {dax_index_ticker} from {start_date_str} to {end_date_str}.")
        return

    print("\nFirst 3 rows of fetched data:")
    print(dax_df.head(3))
    print("\nLast 3 rows of fetched data:")
    print(dax_df.tail(3))
    print(f"\nTotal rows fetched: {len(dax_df)}")

    output_filename_base = f"dax_index_data_{start_date_str}_{end_date_str}"
    output_csv_file = os.path.join(RAW_DIR, f"{output_filename_base}.csv")
    output_parquet_file = os.path.join(RAW_DIR, f"{output_filename_base}.parquet")

    dax_df.to_csv(output_csv_file)
    print(f"\nSaved DAX index data to {output_csv_file}")

    dax_parquet_saved = False
    try:
        dax_df.to_parquet(output_parquet_file)
        print(f"Saved DAX index data to {output_parquet_file} (Parquet format)")
        dax_parquet_saved = True
    except ImportError:
        print("Could not save to Parquet. To enable Parquet, install pyarrow: pip install pyarrow")
    except Exception as e:
        print(f"An error occurred while saving to Parquet: {e}")

    update_manifest(
        start_date=start_date_str,
        end_date=end_date_str,
        dax_price_file=output_parquet_file if dax_parquet_saved else output_csv_file,
        dax_price_rows=len(dax_df),
    )


if __name__ == "__main__":
    main()