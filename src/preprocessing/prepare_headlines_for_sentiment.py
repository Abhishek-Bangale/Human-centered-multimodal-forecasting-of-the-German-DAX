import pandas as pd
import os
import sys
import re
import html # For unescaping HTML entities
from tqdm import tqdm
import csv # For quoting argument if saving to CSV as fallback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import LEADS_DIR, update_manifest

# Optional: For language detection. Install with: pip install langdetect
# try:
#     from langdetect import detect, LangDetectException
#     LANGDETECT_AVAILABLE = True
# except ImportError:
#     LANGDETECT_AVAILABLE = False
#     print("INFO: langdetect library not found. Language detection for NULL language_gdelt entries will be skipped.")
#     print("To install for optional use: pip install langdetect")

# --- Configuration ---
# Repo-root-anchored (data/raw/processed_gdelt_leads), matching where
# process_gdelt_leads.py now writes its output.
INPUT_LEADS_FILE = os.path.join(LEADS_DIR, "dax40_gdelt_article_leads.parquet")
OUTPUT_PREPARED_HEADLINES_FILE = os.path.join(LEADS_DIR, "dax40_prepared_headlines_for_sentiment.parquet")
OUTPUT_DIR = LEADS_DIR # To ensure it exists for the output file

# --- Helper Functions ---
# def detect_language_safe(text): # Example if you want to use langdetect later
#     """
#     Safely detects language of a text snippet.
#     Returns language code (e.g., 'en', 'de') or None if error, too short, or not a string.
#     """
#     if not LANGDETECT_AVAILABLE:
#         return None
#     if not isinstance(text, str) or len(text.strip()) < 15: # Heuristic for min length
#         return None
#     try:
#         return detect(text)
#     except LangDetectException:
#         return "und" # Undetermined, or choose None
#     except Exception:
#         return None

def clean_headline_text(text):
    """Applies basic cleaning to headline text."""
    if not isinstance(text, str):
        return "" # Return empty string for non-strings or NaNs to avoid errors later
    
    cleaned_text = text
    
    # 1. Unescape HTML entities (e.g., & -> &, " -> ")
    cleaned_text = html.unescape(cleaned_text)
    
    # 2. Convert to lowercase (good for consistency before model input)
    cleaned_text = cleaned_text.lower()
    
    # 3. Normalize whitespace (replace multiple spaces/newlines with a single space, strip ends)
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

# --- Main Logic ---
def main_prepare_headlines():
    if not os.path.exists(INPUT_LEADS_FILE):
        print(f"Error: Input GDELT leads file not found: {INPUT_LEADS_FILE}")
        print("Please ensure Script 1 (process_gdelt_leads.py) has run successfully and created this file.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading GDELT leads Parquet file: {INPUT_LEADS_FILE} ...")
    try:
        df = pd.read_parquet(INPUT_LEADS_FILE)
        print(f"Successfully loaded {len(df)} rows from Parquet.")
    except Exception as e:
        print(f"Error loading Parquet file: {e}")
        return

    if df.empty:
        print("The input leads DataFrame is empty. No processing will be done.")
        return

    # 1. Handle Missing/Empty Original Headlines
    print("Step 1: Handling missing or effectively empty original headlines...")
    initial_rows = len(df)
    # Ensure 'original_headline_gdelt_xml' is string type before .str accessor, fill NaNs with empty string
    df['original_headline_gdelt_xml'] = df['original_headline_gdelt_xml'].fillna('').astype(str)
    # Keep rows where the stripped headline is not empty
    df = df[df['original_headline_gdelt_xml'].str.strip() != '']
    
    rows_after_dropping_empty_headlines = len(df)
    print(f"Rows before dropping empty/NaN headlines: {initial_rows}")
    print(f"Rows after dropping empty/NaN headlines: {rows_after_dropping_empty_headlines}")
    
    if df.empty:
        print("No rows left after removing entries with missing/empty headlines. Exiting.")
        return

    # 2. Clean Headline Text
    print("Step 2: Cleaning headline text...")
    tqdm.pandas(desc="Cleaning headlines")
    df['headline_text_for_sentiment'] = df['original_headline_gdelt_xml'].progress_apply(clean_headline_text)
    
    # Drop rows again if cleaning resulted in an empty headline (e.g., if original was just whitespace)
    # Ensure 'headline_text_for_sentiment' is string before .str.strip()
    df['headline_text_for_sentiment'] = df['headline_text_for_sentiment'].fillna('').astype(str)
    df = df[df['headline_text_for_sentiment'].str.strip() != '']
    print(f"Rows after applying cleaning function and removing resulting empty headlines: {len(df)}")
    if df.empty:
        print("No rows left after cleaning headlines. Exiting.")
        return

    # 3. Select and Reorder Columns for the output
    # The 'language_gdelt' column is kept as is from the GDELT processing.
    # The sentiment analysis step will be responsible for filtering by this language.
    print("Step 3: Selecting and reordering final columns...")
    columns_to_keep = [
        'gdelt_event_id',
        'publication_date',
        'publication_datetime_utc',
        'matched_company_ticker',
        'matched_company_canonical_name',
        'source_url',
        'source_name_gdelt',
        'language_gdelt',                 # Original GDELT language, for filtering by sentiment model
        'headline_text_for_sentiment',    # The cleaned headline for analysis
        'original_headline_gdelt_xml',    # Keep original GDELT headline for reference
        'gdelt_tone',
        'gdelt_positive_score',
        'gdelt_negative_score',
        'gdelt_polarity',
        'gdelt_activity_ref_density',     # Added back based on your screenshot
        'gdelt_self_group_ref_density',   # Added back
        'gdelt_themes',                   # Keep for potential feature engineering or context
        'gdelt_locations',
        'gdelt_persons',
        'gdelt_organizations_mentioned_raw'
    ]
    
    # Ensure all selected columns actually exist in the DataFrame to avoid KeyErrors
    existing_columns_to_keep = [col for col in columns_to_keep if col in df.columns]
    prepared_df = df[existing_columns_to_keep].copy() # Use .copy() to avoid SettingWithCopyWarning

    print(f"Final number of prepared headlines for sentiment analysis: {len(prepared_df)}")

    if prepared_df.empty:
        print("No headlines to save after all preparation steps. Output file will not be created.")
        return

    # Save the prepared DataFrame
    try:
        prepared_df.to_parquet(OUTPUT_PREPARED_HEADLINES_FILE, index=False, engine='pyarrow')
        print(f"\nSuccessfully saved prepared headlines to: {OUTPUT_PREPARED_HEADLINES_FILE}")
        update_manifest(prepared_headlines_file=OUTPUT_PREPARED_HEADLINES_FILE, prepared_headlines_rows=len(prepared_df))
    except ImportError:
        # Fallback to CSV if pyarrow not installed, though Parquet is preferred
        output_csv = OUTPUT_PREPARED_HEADLINES_FILE.replace(".parquet", ".csv")
        print(f"\nWarning: pyarrow not installed. Attempting to save to CSV: {output_csv}")
        # Ensure csv library is imported if using its constants like csv.QUOTE_MINIMAL
        prepared_df.to_csv(output_csv, index=False, encoding='utf-8-sig', quoting=csv.QUOTE_MINIMAL)
        print(f"Successfully saved prepared headlines to CSV: {output_csv}")
    except Exception as e:
        print(f"\nError saving prepared headlines data: {type(e).__name__} - {e}")

    print("\nSample of the first 5 rows of prepared headlines data:")
    print(prepared_df.head())
    if not prepared_df.empty:
        print(f"\nColumns in prepared headlines file: {prepared_df.columns.tolist()}")

if __name__ == '__main__':
    # Dummy input file creation for testing this script standalone
    # In real use, `process_gdelt_leads.py` generates the input for this script.
    if not os.path.exists(INPUT_LEADS_FILE):
        print(f"Warning: Input file {INPUT_LEADS_FILE} not found. Creating a dummy one for testing this script.")
        os.makedirs(os.path.dirname(INPUT_LEADS_FILE), exist_ok=True)
        dummy_data = [
            {'gdelt_event_id': 'D1', 'publication_date': '2023-11-30', 'publication_datetime_utc': '2023-11-30 00:00:00', 'matched_company_ticker': 'SAP.DE', 'matched_company_canonical_name': 'SAP SE', 'source_url': 'http://s.com/1', 'source_name_gdelt': 's.com', 'language_gdelt': 'ger', 'original_headline_gdelt_xml': 'SAP gewinnt gro\u00dfen Auftrag & Co.', 'gdelt_themes': 'ECON', 'gdelt_locations': 'DEU', 'gdelt_persons': 'CEO', 'gdelt_organizations_mentioned_raw': 'SAP SE', 'gdelt_tone': 2.5, 'gdelt_positive_score': 3, 'gdelt_negative_score': 0.5, 'gdelt_polarity': 5, 'gdelt_activity_ref_density': 10.0, 'gdelt_self_group_ref_density': 1.0},
            {'gdelt_event_id': 'D2', 'publication_date': '2023-11-30', 'publication_datetime_utc': '2023-11-30 01:00:00', 'matched_company_ticker': 'VOW3.DE', 'matched_company_canonical_name': 'Volkswagen AG', 'source_url': 'http://v.com/1', 'source_name_gdelt': 'v.com', 'language_gdelt': 'eng', 'original_headline_gdelt_xml': '  VW announces NEW EV plans  ', 'gdelt_themes': 'AUTOS', 'gdelt_locations': 'USA', 'gdelt_persons': 'CEO VW', 'gdelt_organizations_mentioned_raw': 'Volkswagen', 'gdelt_tone': 1.0, 'gdelt_positive_score': 2, 'gdelt_negative_score': 1, 'gdelt_polarity': 2, 'gdelt_activity_ref_density': 12.0, 'gdelt_self_group_ref_density': 0.5},
            {'gdelt_event_id': 'D3', 'publication_date': '2023-11-30', 'publication_datetime_utc': '2023-11-30 02:00:00', 'matched_company_ticker': 'ADS.DE', 'matched_company_canonical_name': 'Adidas AG', 'source_url': 'http://a.com/1', 'source_name_gdelt': 'a.com', 'language_gdelt': None, 'original_headline_gdelt_xml': 'Adidas results soon?', 'gdelt_themes': 'SPORTS', 'gdelt_locations': 'DEU', 'gdelt_persons': None, 'gdelt_organizations_mentioned_raw': 'Adidas', 'gdelt_tone': 0, 'gdelt_positive_score': 1, 'gdelt_negative_score': 1, 'gdelt_polarity': 0, 'gdelt_activity_ref_density': 8.0, 'gdelt_self_group_ref_density': 0.0},
            {'gdelt_event_id': 'D4', 'publication_date': '2023-11-30', 'publication_datetime_utc': '2023-11-30 03:00:00', 'matched_company_ticker': 'BAS.DE', 'matched_company_canonical_name': 'BASF SE', 'source_url': 'http://b.com/1', 'source_name_gdelt': 'b.com', 'language_gdelt': 'ger', 'original_headline_gdelt_xml': '    ', 'gdelt_themes': 'CHEM', 'gdelt_locations': 'DEU', 'gdelt_persons': None, 'gdelt_organizations_mentioned_raw': 'BASF', 'gdelt_tone': 0, 'gdelt_positive_score': 0, 'gdelt_negative_score': 0, 'gdelt_polarity': 0, 'gdelt_activity_ref_density': 5.0, 'gdelt_self_group_ref_density': 2.0}, # Empty headline
             {'gdelt_event_id': 'D5', 'publication_date': '2023-12-01', 'publication_datetime_utc': '2023-12-01 00:00:00', 'matched_company_ticker': 'SAP.DE', 'matched_company_canonical_name': 'SAP SE', 'source_url': 'http://s.com/2', 'source_name_gdelt': 's.com', 'language_gdelt': None, 'original_headline_gdelt_xml': None, 'gdelt_themes': 'ECON', 'gdelt_locations': 'DEU', 'gdelt_persons': 'CEO', 'gdelt_organizations_mentioned_raw': 'SAP SE', 'gdelt_tone': 2.5, 'gdelt_positive_score': 3, 'gdelt_negative_score': 0.5, 'gdelt_polarity': 5, 'gdelt_activity_ref_density': 10.0, 'gdelt_self_group_ref_density': 1.0}, # Null headline
        ]
        pd.DataFrame(dummy_data).to_parquet(INPUT_LEADS_FILE, index=False)
        print(f"Created dummy input file: {INPUT_LEADS_FILE}")

    main_prepare_headlines()