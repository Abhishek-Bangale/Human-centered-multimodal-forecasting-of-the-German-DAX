import pandas as pd
import zipfile
import glob
import os
import sys
import re
from tqdm import tqdm
from datetime import datetime
import csv # For handling potential GDELT quoting issues

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import GDELT_DIR, LEADS_DIR, update_manifest

# --- Configuration ---
# DAX 40 Tickers (ensure these are Yahoo Finance compatible, mostly .DE)
# This list should match the tickers used for fetching stock price data.
DAX40_TICKERS_FOR_NEWS_MATCHING = [
    "SAP.DE", "SIE.DE", "ALV.DE", "DTE.DE", "MUV2.DE", "MBG.DE", "AIR.DE", "SHL.DE",
    "IFX.DE", "BAS.DE", "DHL.DE", "DB1.DE", "BMW.DE", "VOW3.DE", "MRK.DE", "ADS.DE",
    "RHM.DE", "EOAN.DE", "BAYN.DE", "DBK.DE", "ENR.DE", "RWE.DE", "HEI.DE", "SRT3.DE",
    "P911.DE", "CBK.DE", "HNR1.DE", "HEN3.DE", "SY1.DE", "VNA.DE", "MTX.DE", "DTG.DE",
    "BEI.DE", "QIA.DE", "PAH3.DE", "FRE.DE", "BNR.DE", "CON.DE", "FME.DE", "ZAL.DE"
]

# GERMAN_COMPANIES_FULL_LIST:
# ***** YOU MUST COMPLETE AND REFINE THIS LIST WITH COMPREHENSIVE ALIASES FOR ALL 40 COMPANIES. *****
# The quality of these aliases directly impacts the number of relevant articles found.
# The 'ticker' value MUST correspond to one from DAX40_TICKERS_FOR_NEWS_MATCHING.
GERMAN_COMPANIES_FULL_LIST = {
    # Canonical Name: {"aliases": ["list", "of", "search", "terms"], "ticker": "TICKER.DE"}
    "SAP SE": {"aliases": ["SAP", "SAP Walldorf", "SAP AG", "S.A.P."], "ticker": "SAP.DE"},
    "Siemens AG": {"aliases": ["Siemens", "Siemens Konzern", "Siemens Group"], "ticker": "SIE.DE"},
    "Allianz SE": {"aliases": ["Allianz", "Allianz Versicherung", "Allianz Group"], "ticker": "ALV.DE"},
    "Volkswagen AG": {"aliases": ["Volkswagen", "VW", "VW Group", "Volkswagen Konzern", "Golf", "Passat", "Audi", "Porsche"], "ticker": "VOW3.DE"},
    "Mercedes-Benz Group AG": {"aliases": ["Mercedes-Benz", "Mercedes", "Daimler AG", "Daimler"], "ticker": "MBG.DE"},
    "Airbus SE": {"aliases": ["Airbus", "Airbus Group", "EADS"], "ticker": "AIR.DE"},
    "BASF SE": {"aliases": ["BASF", "BASF Ludwigshafen", "Badische Anilin- und Sodafabrik"], "ticker": "BAS.DE"},
    "Bayer AG": {"aliases": ["Bayer", "Bayer Leverkusen", "Monsanto"], "ticker": "BAYN.DE"},
    "BMW AG": {"aliases": ["BMW", "Bayerische Motoren Werke", "BMW Group"], "ticker": "BMW.DE"},
    "Deutsche Bank AG": {"aliases": ["Deutsche Bank", "DBK"], "ticker": "DBK.DE"},
    "Deutsche Börse AG": {"aliases": ["Deutsche Börse", "Frankfurt Stock Exchange Operator", "Boerse Frankfurt", "Börse Frankfurt", "Xetra"], "ticker": "DB1.DE"},
    "DHL Group (Deutsche Post AG)": {"aliases": ["Deutsche Post", "DHL", "DHL Group", "Post DHL"], "ticker": "DHL.DE"},
    "Deutsche Telekom AG": {"aliases": ["Deutsche Telekom", "T-Mobile", "Telekom AG", "DTAG", "TCom"], "ticker": "DTE.DE"},
    "E.ON SE": {"aliases": ["E.ON", "EON", "EON SE"], "ticker": "EOAN.DE"},
    "Infineon Technologies AG": {"aliases": ["Infineon", "Infineon Technologies"], "ticker": "IFX.DE"},
    "Münchener Rückversicherungs-Gesellschaft AG (Munich Re)": {"aliases": ["Munich Re", "Münchener Rück", "MUV2", "Muenchener Rueck"], "ticker": "MUV2.DE"},
    "RWE AG": {"aliases": ["RWE", "RWE Group"], "ticker": "RWE.DE"},
    "Siemens Energy AG": {"aliases": ["Siemens Energy", "Siemens Gamesa"], "ticker": "ENR.DE"},
    "Siemens Healthineers AG": {"aliases": ["Siemens Healthineers"], "ticker": "SHL.DE"},
    "Adidas AG": {"aliases": ["Adidas", "Adidas Group"], "ticker": "ADS.DE"},
    "Beiersdorf AG": {"aliases": ["Beiersdorf", "Nivea", "Eucerin", "Labello", "Tesa"], "ticker": "BEI.DE"},
    "Brenntag SE": {"aliases": ["Brenntag"], "ticker": "BNR.DE"},
    "Commerzbank AG": {"aliases": ["Commerzbank", "Coba"], "ticker": "CBK.DE"},
    "Continental AG": {"aliases": ["Continental", "Conti", "Continental Reifen"], "ticker": "CON.DE"},
    "Daimler Truck Holding AG": {"aliases": ["Daimler Truck", "Mercedes-Benz Trucks"], "ticker": "DTG.DE"},
    "Fresenius SE & Co. KGaA": {"aliases": ["Fresenius SE", "Fresenius Group", "Fresenius Kabi", "Fresenius Helios"], "ticker": "FRE.DE"},
    "Fresenius Medical Care AG": {"aliases": ["Fresenius Medical Care", "FMC"], "ticker": "FME.DE"},
    "Hannover Rück SE": {"aliases": ["Hannover Rück", "Hannover Re", "Hannover Rueck"], "ticker": "HNR1.DE"},
    "Heidelberg Materials AG": {"aliases": ["Heidelberg Materials", "HeidelbergCement"], "ticker": "HEI.DE"},
    "Henkel AG & Co. KGaA": {"aliases": ["Henkel", "Persil", "Schwarzkopf", "Loctite", "Pritt", "Dial"], "ticker": "HEN3.DE"},
    "Merck KGaA": {"aliases": ["Merck KGaA", "Merck Darmstadt", "German Merck"], "ticker": "MRK.DE"},
    "MTU Aero Engines AG": {"aliases": ["MTU Aero Engines", "MTU München", "MTU"], "ticker": "MTX.DE"},
    "Dr. Ing. h.c. F. Porsche AG": {"aliases": ["Porsche AG", "Porsche Automobile", "Porsche Cars", "Porsche 911"], "ticker": "P911.DE"},
    "Porsche Automobil Holding SE": {"aliases": ["Porsche SE", "Porsche Holding"], "ticker": "PAH3.DE"},
    "Qiagen N.V.": {"aliases": ["Qiagen"], "ticker": "QIA.DE"},
    "Rheinmetall AG": {"aliases": ["Rheinmetall"], "ticker": "RHM.DE"},
    "Sartorius AG": {"aliases": ["Sartorius", "Sartorius Stedim Biotech"], "ticker": "SRT3.DE"},
    "Symrise AG": {"aliases": ["Symrise"], "ticker": "SY1.DE"},
    "Vonovia SE": {"aliases": ["Vonovia", "Deutsche Annington"], "ticker": "VNA.DE"},
    "Zalando SE": {"aliases": ["Zalando"], "ticker": "ZAL.DE"}
}

# Path to the directory containing downloaded GDELT GKG .CSV.zip files.
# Repo-root-anchored (data/raw/gdelt_gkg_data) to match where
# download_gdelt_gkg.py now actually writes them — the original
# "data/gdelt_gkg_data" was CWD-relative and only happened to line up
# with that script's output if you ran both from the same directory.
GDELT_ZIPS_DIR = GDELT_DIR

# Output directory for processed data (data/raw/processed_gdelt_leads)
OUTPUT_DIR = LEADS_DIR
# Output filename for the GDELT leads (Parquet format recommended)
GDELT_LEADS_FILENAME = "dax40_gdelt_article_leads.parquet"

# GDELT GKG 2.0 Header
GKG_HEADERS = [
    'GKGRECORDID', 'V2.1DATE', 'V2SourceCollectionIdentifier', 'V2SourceCommonName',
    'V2DocumentIdentifier', 'V1Counts', 'V2.1Counts', 'V1Themes', 'V2EnhancedThemes',
    'V1Locations', 'V2EnhancedLocations', 'V1Persons', 'V2EnhancedPersons',
    'V1Organizations', 'V2EnhancedOrganizations', 'V1.5Tone',
    'V2.1EnhancedGKGVisualThemes', 'V2GCAM', 'V2.1SharingImage', 'V2.1RelatedImages',
    'V2.1SocialImageEmbeds', 'V2.1SocialVideoEmbeds', 'V2.1Quotations',
    'V2.1AllNames', 'V2.1Amounts', 'V2.1TranslationInfo', 'V2ExtrasXML'
]
# Columns we are interested in from GDELT for this script
RELEVANT_GKG_COLUMNS = [
    'GKGRECORDID', 'V2.1DATE', 'V2DocumentIdentifier', 'V2SourceCommonName',
    'V2EnhancedOrganizations', 'V2EnhancedThemes', 'V1Locations', 'V1Persons',
    'V1.5Tone', 'V2.1TranslationInfo', 'V2ExtrasXML'
]

# --- Helper Functions ---
def extract_gdelt_language(translation_info_str):
    if pd.isna(translation_info_str) or translation_info_str == "": return None
    match = re.search(r"srclc:([a-zA-Z]{3});", translation_info_str)
    return match.group(1) if match else None

def extract_headline_from_xml(xml_string):
    if pd.isna(xml_string) or xml_string == "": return None
    match = re.search(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", xml_string)
    return match.group(1).strip() if match else None

def parse_tone_string(tone_str):
    if pd.isna(tone_str):
        return {"tone": None, "positive_score": None, "negative_score": None, "polarity": None,
                "activity_ref_density": None, "self_group_ref_density": None}
    parts = tone_str.split(',')
    try:
        return {
            "tone": float(parts[0]), "positive_score": float(parts[1]),
            "negative_score": float(parts[2]), "polarity": float(parts[3]),
            "activity_ref_density": float(parts[4]), "self_group_ref_density": float(parts[5])
        }
    except (IndexError, ValueError):
        return {"tone": None, "positive_score": None, "negative_score": None, "polarity": None,
                "activity_ref_density": None, "self_group_ref_density": None}

def find_matching_companies(text_fields_list, companies_map):
    matched_companies_info = set()
    searchable_text_corpus = " ".join(
        str(field).lower() for field in text_fields_list if pd.notna(field) and str(field).strip() != ""
    )
    if not searchable_text_corpus.strip(): return []

    for canonical_name, details in companies_map.items():
        for alias in details["aliases"]:
            pattern = r'\b' + re.escape(alias.lower()) + r'\b'
            if re.search(pattern, searchable_text_corpus):
                matched_companies_info.add((canonical_name, details["ticker"]))
                break
    return list(matched_companies_info)

# --- Main Processing Logic ---
def process_gdelt_files_for_leads():
    if not os.path.exists(GDELT_ZIPS_DIR):
        print(f"Error: GDELT Zips directory '{GDELT_ZIPS_DIR}' not found.")
        print("Please create it and place your GDELT GKG .csv.zip files inside.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Enhanced glob patterns to find GDELT files
    patterns_to_check = [
        "*.gkg.csv.zip",      # Common GDELT 2.0 GKG
        "*.gkg.CSV.zip",      # Case variation
        "*.export.CSV.zip",   # GDELT 1.0 GKG (often named this way)
        "*.csv.zip"           # Generic, as a fallback if specific patterns fail
    ]
    
    zip_files_found = []
    for pattern in patterns_to_check:
        zip_files_found.extend(glob.glob(os.path.join(GDELT_ZIPS_DIR, pattern)))
    
    # Remove duplicates if patterns overlap
    zip_files_to_process_list = sorted(list(set(zip_files_found)))

    # Heuristic filtering to prioritize likely GKG files if many types are present
    if any(".gkg.csv.zip" in f.lower() for f in zip_files_to_process_list):
        zip_files_to_process_list = [f for f in zip_files_to_process_list if ".gkg.csv.zip" in f.lower()]
    elif any(".export.csv.zip" in f.lower() for f in zip_files_to_process_list): # Check for GDELT 1.0 type GKG
         # This check is imperfect, as Events/Mentions also use .export.CSV.zip
         # A more robust check would be to inspect a few lines of the CSV inside.
         # For now, assume if it's .export.CSV.zip and not explicitly events/mentions, it might be GKG.
        zip_files_to_process_list = [f for f in zip_files_to_process_list if ".export.csv.zip" in f.lower()]


    if not zip_files_to_process_list:
        print(f"No suitable GDELT GKG zip files found in '{GDELT_ZIPS_DIR}' using known patterns.")
        return
    
    print(f"Found {len(zip_files_to_process_list)} GDELT GKG-like zip files to process in '{GDELT_ZIPS_DIR}'.")

    all_article_leads = []
    unique_lead_tracker = set()

    # zip_files_to_process_sorted = zip_files_to_process_list[:1] # For quick testing: process only the first file
    zip_files_to_process_sorted = zip_files_to_process_list # Process all found files

    for zip_filepath in tqdm(zip_files_to_process_sorted, desc="Processing GDELT files"):
        try:
            with zipfile.ZipFile(zip_filepath, 'r') as zf:
                csv_candidates = [name for name in zf.namelist() if name.lower().endswith('.csv')]
                if not csv_candidates:
                    print(f"Warning: No .csv file found in zip: {zip_filepath}. Skipping.")
                    continue
                csv_filename_in_zip = csv_candidates[0] 

                with zf.open(csv_filename_in_zip, 'r') as f:
                    try:
                        # Attempt to read with specified dtypes for potentially problematic columns
                        # This is a guess; actual problematic columns may vary.
                        # GDELT fields are mostly strings or numbers that pandas can infer,
                        # but sometimes specific columns can cause issues if very mixed.
                        # For now, let pandas infer, and add specific dtypes if errors arise.
                        df_chunk = pd.read_csv(f, sep='\t', header=None, names=GKG_HEADERS,
                                               usecols=RELEVANT_GKG_COLUMNS,
                                               encoding='utf-8', quoting=csv.QUOTE_NONE,
                                               on_bad_lines='warn', low_memory=False)
                    except UnicodeDecodeError:
                        f.seek(0)
                        df_chunk = pd.read_csv(f, sep='\t', header=None, names=GKG_HEADERS,
                                               usecols=RELEVANT_GKG_COLUMNS,
                                               encoding='latin1', quoting=csv.QUOTE_NONE,
                                               on_bad_lines='warn', low_memory=False)
            
            for _, row in df_chunk.iterrows():
                gdelt_orgs = str(row.get('V2EnhancedOrganizations', ''))
                gdelt_themes = str(row.get('V2EnhancedThemes', ''))
                xml_headline = extract_headline_from_xml(row.get('V2ExtrasXML', ''))
                
                search_corpus = [gdelt_orgs, gdelt_themes, xml_headline]
                matched_companies = find_matching_companies(search_corpus, GERMAN_COMPANIES_FULL_LIST)

                if matched_companies:
                    article_url = str(row.get('V2DocumentIdentifier', ''))
                    if not article_url or not (article_url.startswith('http://') or article_url.startswith('https://')):
                        continue

                    publication_datetime_gkg = str(row.get('V2.1DATE', ''))
                    pub_date_str, pub_datetime_utc_str = None, None
                    try:
                        dt_obj = datetime.strptime(publication_datetime_gkg, '%Y%m%d%H%M%S')
                        pub_date_str = dt_obj.strftime('%Y-%m-%d')
                        pub_datetime_utc_str = dt_obj.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        try:
                            dt_obj = datetime.strptime(publication_datetime_gkg[:8], '%Y%m%d')
                            pub_date_str = dt_obj.strftime('%Y-%m-%d')
                            pub_datetime_utc_str = dt_obj.strftime('%Y-%m-%d 00:00:00')
                        except ValueError:
                            continue
                    
                    gdelt_lang = extract_gdelt_language(row.get('V2.1TranslationInfo', ''))
                    tone_data = parse_tone_string(row.get('V1.5Tone', ''))

                    for company_name, ticker in matched_companies:
                        lead_key = (article_url, ticker, pub_date_str)
                        if lead_key in unique_lead_tracker:
                            continue
                        
                        lead_data = {
                            'gdelt_event_id': str(row.get('GKGRECORDID', '')),
                            'publication_date': pub_date_str,
                            'publication_datetime_utc': pub_datetime_utc_str,
                            'matched_company_ticker': ticker,
                            'matched_company_canonical_name': company_name,
                            'source_url': article_url,
                            'source_name_gdelt': str(row.get('V2SourceCommonName', '')),
                            'language_gdelt': gdelt_lang,
                            'original_headline_gdelt_xml': xml_headline,
                            'gdelt_themes': gdelt_themes,
                            'gdelt_locations': str(row.get('V1Locations', '')),
                            'gdelt_persons': str(row.get('V1Persons', '')),
                            'gdelt_organizations_mentioned_raw': gdelt_orgs,
                            'gdelt_tone': tone_data['tone'],
                            'gdelt_positive_score': tone_data['positive_score'],
                            'gdelt_negative_score': tone_data['negative_score'],
                            'gdelt_polarity': tone_data['polarity'],
                            'gdelt_activity_ref_density': tone_data['activity_ref_density'],
                            'gdelt_self_group_ref_density': tone_data['self_group_ref_density']
                        }
                        all_article_leads.append(lead_data)
                        unique_lead_tracker.add(lead_key)
        
        except pd.errors.EmptyDataError:
            print(f"Warning: GDELT file {zip_filepath} (or CSV inside) was empty or unparseable. Skipping.")
        except zipfile.BadZipFile:
            print(f"Warning: File {zip_filepath} is not a valid zip file or is corrupted. Skipping.")
        except Exception as e:
            print(f"Major error processing GDELT zip file {zip_filepath}: {type(e).__name__} - {e}")

    if not all_article_leads:
        print("No relevant article leads found after processing all GDELT files.")
        return

    leads_df = pd.DataFrame(all_article_leads)
    
    if leads_df.empty:
        print("DataFrame of leads is empty. No output file will be generated.")
        return
        
    leads_df.drop_duplicates(subset=['source_url', 'matched_company_ticker', 'publication_date'], 
                              keep='first', inplace=True)

    output_parquet_path = os.path.join(OUTPUT_DIR, GDELT_LEADS_FILENAME)
    try:
        leads_df.to_parquet(output_parquet_path, index=False, engine='pyarrow')
        print(f"\nSuccessfully saved GDELT article leads to: {output_parquet_path}")
        print(f"Total unique article leads extracted: {len(leads_df)}")
        update_manifest(gdelt_leads_file=output_parquet_path, gdelt_leads_rows=len(leads_df))
    except ImportError:
        print("\nError: pyarrow library not found. Parquet output failed. Please install with: pip install pyarrow")
    except Exception as e:
        print(f"\nError saving leads data to Parquet {output_parquet_path}: {e}")

    print("\nSample of the first 5 GDELT article leads:")
    print(leads_df.head())
    if not leads_df.empty:
        print(f"\nColumns in the GDELT leads Parquet file: {leads_df.columns.tolist()}")

if __name__ == '__main__':
    # Create dummy GDELT directory and a sample file if they don't exist for testing
    # This helps test the script logic even without actual GDELT downloads.
    if not os.path.exists(GDELT_ZIPS_DIR):
        print(f"GDELT Zips directory '{GDELT_ZIPS_DIR}' not found. Creating dummy data for testing.")
        os.makedirs(GDELT_ZIPS_DIR, exist_ok=True)
        # Create a dummy zip file with a couple of GKG-like lines
        dummy_zip_path = os.path.join(GDELT_ZIPS_DIR, "dummy_20231130000000.gkg.csv.zip")
        if not os.path.exists(dummy_zip_path):
            dummy_content_lines = [
                "DUMMYID1\t20231130080000\thttp://test.com/news_sap_1\ttestnews.com\tSAP SE;Microsoft\tECON_STOCKMARKET;TECH_AI\tDEU;BERLIN;USA\tChristian Klein;Satya Nadella\t-1.5,2.1,3.6,0.8,10.0,1.0\tsrclc:ger;confidence:0.95;\t<PAGE_TITLE>SAP Partners with Microsoft on AI</PAGE_TITLE>",
                "DUMMYID2\t20231130090000\thttp://test.com/news_vw_1\ttestauto.com\tVolkswagen AG;Tesla Inc.\tMANUF_AUTOS;TECH_EV\tDEU;WOLFSBURG;USA\tOliver Blume;Elon Musk\t2.0,4.0,2.0,5.0,12.0,0.5\tsrclc:eng;confidence:0.88;\t<PAGE_TITLE>Volkswagen Challenges Tesla with New EV</PAGE_TITLE>",
                "DUMMYID_NO_MATCH\t20231130100000\thttp://test.com/other_news\tothernews.com\tSomeCompany\tGENERAL_NEWS\tUSA\t\t1.0,1.0,0.0,1.0,5.0,0.0\tsrclc:eng;\t<PAGE_TITLE>Other News Story</PAGE_TITLE>"
            ]
            dummy_csv_content = "\n".join(dummy_content_lines)
            with zipfile.ZipFile(dummy_zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                zf.writestr("dummy_20231130000000.gkg.csv", dummy_csv_content.encode('utf-8'))
            print(f"Created dummy GDELT file: {dummy_zip_path}")
    
    process_gdelt_files_for_leads() 