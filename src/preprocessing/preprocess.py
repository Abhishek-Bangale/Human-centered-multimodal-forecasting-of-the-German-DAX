import os
import warnings

# Suppress TensorFlow warnings and oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional, Literal
import logging
import json # Though not explicitly used in provided snippet, good for config/complex data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import REPO_ROOT, DATA_DIR, load_manifest, StageTimer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DAXNewsSentimentAnalyzer:
    def __init__(self, model_name: str = "ProsusAI/finbert", use_gpu: bool = True):
        logger.info(f"Initializing sentiment analyzer with model: {model_name}...")
        
        self.model_name = model_name # Allow different models
        self.device = self._setup_device(use_gpu)
        self.tokenizer = None
        self.model = None
        
        self._load_model()
        
        # Label mapping might change based on the model
        if self.model_name == "ProsusAI/finbert":
            self.label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}
        elif self.model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
            # This model outputs "1 star", "2 stars", etc.
            # We'll handle mapping these to positive/negative/neutral later
            self.label_mapping = { # Index to Star Label (example, might vary)
                0: '1 star', 1: '2 stars', 2: '3 stars', 3: '4 stars', 4: '5 stars'
            }
        else:
            # Default or raise error if unknown model with specific mapping
            logger.warning(f"No specific label mapping for {self.model_name}. Assuming standard classification output.")
            # A common default might be 0: negative, 1: neutral, 2: positive (needs verification per model)
            self.label_mapping = {i: f"label_{i}" for i in range(self.model.config.num_labels)}


        logger.info(f"Sentiment analyzer model '{self.model_name}' loaded successfully on {self.device}!")
    
    def _setup_device(self, use_gpu: bool) -> torch.device:
        if use_gpu and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif use_gpu and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): # For Apple Silicon
            device = torch.device("mps")
            logger.info("Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU for inference")
        return device
    
    def _load_model(self):
        try:
            logger.info(f"Loading tokenizer for {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            logger.info(f"Loading model {self.model_name}...")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer '{self.model_name}': {str(e)}")
            raise RuntimeError(f"Could not load model/tokenizer: {str(e)}")

    def _map_stars_to_sentiment_scores(self, label: str, score: float) -> Dict[str, float]:
        """Maps nlptown star labels to positive/negative/neutral probabilities."""
        # This is a way to get distinct prob-like scores if model gives class & confidence
        if "star" in label: # Specific to nlptown model
            if label in ["1 star", "2 stars"]:
                return {'positive': 0.05, 'negative': score, 'neutral': 0.95 - score} # Heuristic
            elif label == "3 stars":
                return {'positive': 0.1 * score, 'negative': 0.1 * score, 'neutral': score} # Heuristic
            elif label in ["4 stars", "5 stars"]:
                return {'positive': score, 'negative': 0.05, 'neutral': 0.95 - score} # Heuristic
        return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} # Default if not stars


    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        if not text or pd.isna(text) or not isinstance(text, str) or text.strip() == "":
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} # Default for empty
        
        try:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            if self.model_name == "nlptown/bert-base-multilingual-uncased-sentiment":
                # This model's output needs specific handling
                # The pipeline does this: logits -> argmax -> label
                # For probabilities, we apply softmax to logits
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
                predicted_class_id = np.argmax(probs)
                label = self.model.config.id2label[predicted_class_id] # Gets "1 star", "2 stars" etc.
                score = probs[predicted_class_id] # Confidence in that star rating
                # Map star rating to our positive/negative/neutral schema
                # This is a heuristic mapping, can be refined.
                if label == "1 star": sentiment_scores = {'positive':0.0, 'negative':0.8, 'neutral':0.2}
                elif label == "2 stars": sentiment_scores = {'positive':0.1, 'negative':0.6, 'neutral':0.3}
                elif label == "3 stars": sentiment_scores = {'positive':0.3, 'negative':0.3, 'neutral':0.4}
                elif label == "4 stars": sentiment_scores = {'positive':0.6, 'negative':0.1, 'neutral':0.3}
                elif label == "5 stars": sentiment_scores = {'positive':0.8, 'negative':0.0, 'neutral':0.2}
                else: sentiment_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} # Fallback

            elif self.model_name == "ProsusAI/finbert": # Or other models that output 3-class logits for pos/neg/neu
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = predictions.cpu().numpy()[0]
                sentiment_scores = {}
                for idx, prob_val in enumerate(probs):
                    label = self.label_mapping[idx]
                    sentiment_scores[label] = float(prob_val)
            else: # Generic approach for other models, assuming logits for num_labels
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = predictions.cpu().numpy()[0]
                sentiment_scores = {}
                # This assumes labels are ordered if self.label_mapping is just based on index
                for idx, prob_val in enumerate(probs):
                    label = self.label_mapping.get(idx, f"label_{idx}")
                    sentiment_scores[label] = float(prob_val)
                # Ensure positive, negative, neutral keys exist even if model output is different
                sentiment_scores.setdefault('positive', 0.0)
                sentiment_scores.setdefault('negative', 0.0)
                sentiment_scores.setdefault('neutral', 1.0 - sentiment_scores.get('positive',0) - sentiment_scores.get('negative',0))


            return sentiment_scores
            
        except Exception as e:
            logger.warning(f"Error analyzing sentiment for text snippet '{str(text)[:50]}...': {type(e).__name__} - {str(e)}")
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} # Default on error
    
    def analyze_headlines_batch(self, headlines: List[str], languages: Optional[List[str]] = None, batch_size: int = 16) -> pd.DataFrame:
        sentiment_data = []
        total_headlines = len(headlines)
        
        logger.info(f"Starting sentiment analysis for {total_headlines} headlines...")
        
        for i in range(0, total_headlines, batch_size):
            batch_headlines = headlines[i:i + batch_size]
            batch_languages = languages[i:i + batch_size] if languages else [None] * len(batch_headlines)

            for j, headline_text in enumerate(batch_headlines):
                current_lang = batch_languages[j]
                
                # Language filtering specific for ProsusAI/finbert
                if self.model_name == "ProsusAI/finbert" and current_lang and current_lang != 'eng':
                    # logger.debug(f"Skipping non-English headline for ProsusAI/finbert: '{headline_text[:30]}...' (lang: {current_lang})")
                    sentiment_scores = {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34} # Default for non-English
                else:
                    sentiment_scores = self.analyze_sentiment(headline_text)
                
                sentiment_data.append({
                    'headline': headline_text, # Original headline from the batch
                    'positive_score': sentiment_scores['positive'],
                    'negative_score': sentiment_scores['negative'],
                    'neutral_score': sentiment_scores['neutral'],
                    'headline_sentiment_score': sentiment_scores['positive'] - sentiment_scores['negative'],
                    'dominant_sentiment': max(sentiment_scores, key=sentiment_scores.get)
                })
            
            processed = min(i + batch_size, total_headlines)
            if processed % (batch_size * 10) == 0 or processed == total_headlines : # Log less frequently
                 logger.info(f"Processed {processed}/{total_headlines} headlines ({processed/total_headlines*100:.1f}%)")
        
        return pd.DataFrame(sentiment_data)

    # ... [aggregate_daily_sentiment, _get_default_sentiment functions remain largely the same as you provided] ...
    # Minor changes: ensure float conversion for safety if numpy types persist.
    def aggregate_daily_sentiment(self, sentiment_df: pd.DataFrame, 
                                aggregation_method: str = 'mean',
                                include_headline_details: bool = False) -> Dict:
        if sentiment_df.empty:
            return self._get_default_sentiment(aggregation_method, include_headline_details)
        
        if aggregation_method == 'mean':
            avg_positive = sentiment_df['positive_score'].mean()
            avg_negative = sentiment_df['negative_score'].mean()
            avg_neutral = sentiment_df['neutral_score'].mean()
        elif aggregation_method == 'weighted':
            weights = 1 - sentiment_df['neutral_score']
            weights_sum = weights.sum()
            if weights_sum > 0: weights = weights / weights_sum
            else: weights = np.ones(len(sentiment_df)) / len(sentiment_df) if len(sentiment_df) > 0 else np.array([])
            
            if len(weights) > 0:
                avg_positive = (sentiment_df['positive_score'] * weights).sum()
                avg_negative = (sentiment_df['negative_score'] * weights).sum()
                avg_neutral = (sentiment_df['neutral_score'] * weights).sum()
            else: # Should not happen if sentiment_df is not empty, but as a safeguard
                return self._get_default_sentiment(aggregation_method, include_headline_details)

        elif aggregation_method == 'max_impact':
            if sentiment_df.empty or 'headline_sentiment_score' not in sentiment_df.columns:
                 return self._get_default_sentiment(aggregation_method, include_headline_details)
            max_impact_idx = sentiment_df['headline_sentiment_score'].abs().idxmax()
            max_impact_row = sentiment_df.loc[max_impact_idx]
            avg_positive = max_impact_row['positive_score']
            avg_negative = max_impact_row['negative_score']
            avg_neutral = max_impact_row['neutral_score']
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        sentiment_scores_dict = {'positive': avg_positive, 'negative': avg_negative, 'neutral': avg_neutral}
        dominant_sentiment = max(sentiment_scores_dict, key=sentiment_scores_dict.get)
        sentiment_score = avg_positive - avg_negative
        sentiment_intensity = 1 - avg_neutral
        sentiment_volatility = sentiment_df['headline_sentiment_score'].std()
        
        result = {
            'avg_positive': float(avg_positive), 'avg_negative': float(avg_negative),
            'avg_neutral': float(avg_neutral), 'dominant_sentiment': dominant_sentiment,
            'sentiment_score': float(sentiment_score), 'sentiment_intensity': float(sentiment_intensity),
            'sentiment_volatility': float(sentiment_volatility) if not pd.isna(sentiment_volatility) else 0.0,
            'num_headlines': len(sentiment_df),
            'positive_headlines_count': int((sentiment_df['dominant_sentiment'] == 'positive').sum()),
            'negative_headlines_count': int((sentiment_df['dominant_sentiment'] == 'negative').sum()),
            'neutral_headlines_count': int((sentiment_df['dominant_sentiment'] == 'neutral').sum()),
            'aggregation_method': aggregation_method,
            'max_sentiment_score': float(sentiment_df['headline_sentiment_score'].max() if not sentiment_df.empty else 0.0),
            'min_sentiment_score': float(sentiment_df['headline_sentiment_score'].min() if not sentiment_df.empty else 0.0)
        }
        if include_headline_details:
            result['headline_details'] = sentiment_df.to_dict(orient='records')
        return result

    def _get_default_sentiment(self, aggregation_method: str = 'mean', include_headline_details: bool = False) -> Dict:
        result = {
            'avg_positive': 0.33, 'avg_negative': 0.33, 'avg_neutral': 0.34,
            'dominant_sentiment': 'neutral', 'sentiment_score': 0.0, 'sentiment_intensity': 0.0,
            'sentiment_volatility': 0.0, 'num_headlines': 0, 'positive_headlines_count': 0,
            'negative_headlines_count': 0, 'neutral_headlines_count': 0,
            'aggregation_method': aggregation_method, 'max_sentiment_score': 0.0, 'min_sentiment_score': 0.0
        }
        if include_headline_details: result['headline_details'] = []
        return result

    def precompute_sentiment_cache(self, news_df: pd.DataFrame, news_headline_col: str,
                                    news_lang_col: Optional[str] = None) -> pd.DataFrame:
        """
        Scores every unique headline exactly ONCE.

        The original create_hybrid_dataset() called create_granular_dataset()
        (which scores each day's headlines) and then create_aggregated_dataset()
        three separate times, once per aggregation method (mean/weighted/
        max_impact) — each of those ALSO independently re-scored the same
        underlying headlines. That's up to 4x redundant transformer inference
        over an identical set of headlines, which was the single biggest lever
        on this pipeline's runtime (see README).

        Sentiment score is a pure function of (headline text, language) for
        this analyzer — nothing about which day-window a headline belongs to
        changes its score — so scoring each unique (headline, language) pair
        once and looking it up everywhere else is exactly equivalent, just
        without the repeated work.
        """
        logger.info("Pre-computing sentiment for every unique headline (scored once, reused for all outputs)...")
        key_cols = [news_headline_col] + ([news_lang_col] if news_lang_col and news_lang_col in news_df.columns else [])
        unique_df = news_df[key_cols].dropna(subset=[news_headline_col]).drop_duplicates().reset_index(drop=True)

        headlines_list = unique_df[news_headline_col].tolist()
        languages_list = unique_df[news_lang_col].tolist() if news_lang_col and news_lang_col in unique_df.columns else None

        sentiment_results_df = self.analyze_headlines_batch(headlines_list, languages=languages_list).reset_index(drop=True)
        cache_df = pd.concat([unique_df, sentiment_results_df.drop(columns=['headline'])], axis=1)
        logger.info(f"Sentiment cache built for {len(cache_df):,} unique headlines "
                    f"(vs. re-scoring per day/method as the original code did).")
        return cache_df

    def _lookup_or_score(self, headlines_list: List[str], languages_list: Optional[List[str]],
                          sentiment_cache: Optional[pd.DataFrame], news_headline_col: str,
                          news_lang_col: Optional[str]) -> pd.DataFrame:
        """Returns a DataFrame in the exact schema analyze_headlines_batch()
        produces (one row per input headline, same order, duplicates
        preserved) — either by a fast merge against a precomputed cache, or
        by actually scoring on the fly if no cache was supplied (keeps
        create_granular_dataset/create_aggregated_dataset usable standalone,
        matching their original behavior, when someone calls them directly
        without going through create_hybrid_dataset)."""
        if sentiment_cache is None or not headlines_list:
            return self.analyze_headlines_batch(headlines_list, languages=languages_list)

        key_cols = [news_headline_col] + ([news_lang_col] if news_lang_col and news_lang_col in sentiment_cache.columns else [])
        local_df = pd.DataFrame({news_headline_col: headlines_list})
        if news_lang_col and news_lang_col in sentiment_cache.columns:
            local_df[news_lang_col] = languages_list if languages_list is not None else [None] * len(headlines_list)

        merged = local_df.merge(sentiment_cache, on=key_cols, how='left')
        merged = merged.rename(columns={news_headline_col: 'headline'})
        score_cols = ['positive_score', 'negative_score', 'neutral_score', 'headline_sentiment_score', 'dominant_sentiment']
        missing = merged[score_cols].isna().any(axis=1)
        if missing.any():
            # Shouldn't normally happen (cache is built from the same news_df),
            # but fall back to on-the-fly scoring for any cache miss so this
            # never silently drops or mis-scores a headline.
            fallback = self.analyze_headlines_batch(
                [h for h in merged.loc[missing, 'headline']],
                languages=[languages_list[i] for i in merged.index[missing]] if languages_list else None
            )
            merged.loc[missing, score_cols] = fallback[score_cols].values
        return merged[['headline'] + score_cols]

    def create_granular_dataset(self, dax_df: pd.DataFrame, news_df: pd.DataFrame,
                               dax_date_col: str, news_date_col: str, news_headline_col: str,
                               news_lang_col: Optional[str] = None,
                               sentiment_cache: Optional[pd.DataFrame] = None) -> pd.DataFrame: # Added news_lang_col
        logger.info("Creating granular dataset with individual headlines...")
        granular_data = []
        news_dates = sorted(news_df[news_date_col].dt.date.unique())
        total_dates = len(news_dates)
        logger.info(f"Processing {total_dates} unique news dates for granular dataset...")

        for idx, news_date_val in enumerate(news_dates):
            next_dax_date = news_date_val + timedelta(days=1)
            news_day_df = news_df[news_df[news_date_col].dt.date == news_date_val]
            dax_day_df = dax_df[dax_df[dax_date_col].dt.date == next_dax_date]

            if not news_day_df.empty and not dax_day_df.empty:
                headlines_list = news_day_df[news_headline_col].dropna().tolist()
                languages_list = news_day_df[news_lang_col].tolist() if news_lang_col and news_lang_col in news_day_df else None
                
                if headlines_list:
                    sentiment_results_df = self._lookup_or_score(
                        headlines_list, languages_list, sentiment_cache, news_headline_col, news_lang_col
                    )

                    for _, sentiment_row in sentiment_results_df.iterrows():
                        dax_row_data = dax_day_df.iloc[0] # Assuming one DAX entry for next day
                        
                        granular_entry = dax_row_data.to_dict()
                        granular_entry['news_date'] = news_date_val # Date of the news
                        # Use original headline from news_day_df that matches sentiment_row['headline']
                        # This requires careful matching if headlines are not unique.
                        # A safer way would be to join sentiment_results_df back to news_day_df.
                        # For simplicity here, assuming 'headline' in sentiment_row is sufficient.
                        original_news_row = news_day_df[news_day_df[news_headline_col] == sentiment_row['headline']].iloc[0]
                        
                        granular_entry['headline_text_analyzed'] = sentiment_row['headline']
                        granular_entry['source_url'] = original_news_row.get('source_url', None) # Get other details
                        granular_entry['matched_company_ticker'] = original_news_row.get('matched_company_ticker', None)
                        granular_entry['language_gdelt'] = original_news_row.get(news_lang_col if news_lang_col else 'language_gdelt', None)

                        granular_entry['headline_positive_score'] = sentiment_row['positive_score']
                        granular_entry['headline_negative_score'] = sentiment_row['negative_score']
                        granular_entry['headline_neutral_score'] = sentiment_row['neutral_score']
                        granular_entry['headline_sentiment_score'] = sentiment_row['headline_sentiment_score']
                        granular_entry['headline_dominant_sentiment'] = sentiment_row['dominant_sentiment']
                        granular_data.append(granular_entry)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_dates: # Log less frequently
                logger.info(f"Granular processed {idx + 1}/{total_dates} news dates ({(idx + 1)/total_dates*100:.1f}%)")
        
        logger.info("Granular dataset creation completed!")
        return pd.DataFrame(granular_data)
    
    def create_aggregated_dataset(self, dax_df: pd.DataFrame, news_df: pd.DataFrame,
                                dax_date_col: str, news_date_col: str, news_headline_col: str,
                                news_lang_col: Optional[str] = None, # Added news_lang_col
                                aggregation_method: str = 'mean',
                                include_headline_details: bool = False,
                                sentiment_cache: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        logger.info(f"Creating aggregated dataset using '{aggregation_method}' method...")
        matched_data = []
        dax_unique_dates = sorted(dax_df[dax_date_col].dt.date.unique())
        total_dax_dates = len(dax_unique_dates)
        logger.info(f"Processing {total_dax_dates} unique DAX dates for aggregation...")

        for idx, current_dax_date in enumerate(dax_unique_dates):
            prev_news_date = current_dax_date - timedelta(days=1)
            current_dax_day_data = dax_df[dax_df[dax_date_col].dt.date == current_dax_date]
            prev_day_news_data = news_df[news_df[news_date_col].dt.date == prev_news_date]
            
            daily_sentiment_agg = {}
            if not prev_day_news_data.empty:
                headlines_list = prev_day_news_data[news_headline_col].dropna().tolist()
                languages_list = prev_day_news_data[news_lang_col].tolist() if news_lang_col and news_lang_col in prev_day_news_data else None
                
                if headlines_list:
                    sentiment_results_df = self._lookup_or_score(
                        headlines_list, languages_list, sentiment_cache, news_headline_col, news_lang_col
                    )
                    daily_sentiment_agg = self.aggregate_daily_sentiment(
                        sentiment_results_df, aggregation_method, include_headline_details
                    )
                else:
                    daily_sentiment_agg = self._get_default_sentiment(aggregation_method, include_headline_details)
            else:
                daily_sentiment_agg = self._get_default_sentiment(aggregation_method, include_headline_details)
            
            for _, dax_row_iter in current_dax_day_data.iterrows():
                matched_entry = dax_row_iter.to_dict()
                matched_entry['sentiment_date'] = prev_news_date # Date of the news used for sentiment
                matched_entry.update(daily_sentiment_agg)
                matched_data.append(matched_entry)
            
            if (idx + 1) % 50 == 0 or (idx + 1) == total_dax_dates: # Log less frequently
                logger.info(f"Aggregated processed {idx + 1}/{total_dax_dates} DAX dates ({(idx + 1)/total_dax_dates*100:.1f}%)")
        
        logger.info("Aggregated dataset creation completed!")
        return pd.DataFrame(matched_data)

    def create_hybrid_dataset(self, dax_df: pd.DataFrame, news_df: pd.DataFrame,
                            dax_date_col: str, news_date_col: str, news_headline_col: str,
                            news_lang_col: Optional[str] = None) -> Tuple[Optional[pd.DataFrame], Dict[str, pd.DataFrame]]:
        logger.info("Creating hybrid dataset (both granular and aggregated types)...")

        # Score every unique headline ONCE up front instead of letting each
        # of the 4 datasets below (granular + 3 aggregation methods) redo
        # the same transformer inference independently — see
        # precompute_sentiment_cache()'s docstring for why this is safe.
        sentiment_cache = self.precompute_sentiment_cache(news_df, news_headline_col, news_lang_col)

        granular_output_df = self.create_granular_dataset(
            dax_df, news_df, dax_date_col, news_date_col, news_headline_col, news_lang_col,
            sentiment_cache=sentiment_cache
        )

        aggregated_output_dfs = {}
        for agg_method in ['mean', 'weighted', 'max_impact']:
            logger.info(f"Generating aggregated dataset for method: {agg_method}")
            df = self.create_aggregated_dataset(
                dax_df, news_df, dax_date_col, news_date_col, news_headline_col, news_lang_col,
                aggregation_method=agg_method, include_headline_details=True, # Keep details for potential UI use
                sentiment_cache=sentiment_cache
            )
            aggregated_output_dfs[agg_method] = df

        return granular_output_df, aggregated_output_dfs
    
    def load_and_prepare_data(self, dax_file: str, news_file: str, 
                            dax_date_col: str, news_date_col: str,
                            news_headline_col: str, news_lang_col: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading DAX data from: {dax_file}")
        logger.info(f"Loading News data from: {news_file}")
        try:
            if dax_file.endswith('.parquet'): dax_df = pd.read_parquet(dax_file)
            elif dax_file.endswith('.csv'): dax_df = pd.read_csv(dax_file)
            elif dax_file.endswith('.xlsx'): dax_df = pd.read_excel(dax_file)
            else: raise ValueError("DAX file must be Parquet, CSV, or Excel")
        except Exception as e: raise FileNotFoundError(f"Could not load DAX file '{dax_file}': {e}")

        try:
            if news_file.endswith('.parquet'): news_df = pd.read_parquet(news_file)
            elif news_file.endswith('.csv'): news_df = pd.read_csv(news_file)
            elif news_file.endswith('.xlsx'): news_df = pd.read_excel(news_file)
            else: raise ValueError("News file must be Parquet, CSV, or Excel")
        except Exception as e: raise FileNotFoundError(f"Could not load News file '{news_file}': {e}")
        
        # Validate columns
        if dax_date_col not in dax_df.columns:
            # If date is index, reset it
            if dax_df.index.name and dax_df.index.name.lower() == dax_date_col.lower():
                dax_df = dax_df.reset_index()
            else:
                raise ValueError(f"Date column '{dax_date_col}' not found in DAX data. Columns: {dax_df.columns.tolist()}")
        
        for col in [news_date_col, news_headline_col]:
            if col not in news_df.columns:
                raise ValueError(f"Column '{col}' not found in News data. Columns: {news_df.columns.tolist()}")
        if news_lang_col and news_lang_col not in news_df.columns:
             logger.warning(f"Language column '{news_lang_col}' not specified or not found in News data. Language-specific filtering in analyzer might be affected.")


        dax_df[dax_date_col] = pd.to_datetime(dax_df[dax_date_col])
        news_df[news_date_col] = pd.to_datetime(news_df[news_date_col])
        
        dax_df = dax_df.sort_values(by=dax_date_col).reset_index(drop=True)
        news_df = news_df.sort_values(by=news_date_col).reset_index(drop=True)
        
        logger.info(f"Loaded {len(dax_df):,} DAX records ({dax_df[dax_date_col].min().date()} to {dax_df[dax_date_col].max().date()})")
        logger.info(f"Loaded {len(news_df):,} News records ({news_df[news_date_col].min().date()} to {news_df[news_date_col].max().date()})")
        
        if not news_df.empty:
            headlines_per_date = news_df.groupby(news_df[news_date_col].dt.date).size()
            logger.info(f"News headlines per date - Mean: {headlines_per_date.mean():.1f}, Max: {headlines_per_date.max()}, Min: {headlines_per_date.min()}")
        
        return dax_df, news_df

    def save_datasets(self, datasets: Dict[str, Optional[pd.DataFrame]], base_output_path: str):
        os.makedirs(os.path.dirname(base_output_path) or '.', exist_ok=True)
        for name, df_to_save in datasets.items():
            if df_to_save is None or df_to_save.empty:
                logger.info(f"Dataset '{name}' is empty, skipping save.")
                continue

            # Construct filename
            path_prefix, extension = os.path.splitext(base_output_path)
            if not extension: # If no extension, assume .parquet
                extension = ".parquet"
            
            output_file = f"{path_prefix}_{name}{extension}"
            
            try:
                if extension == '.parquet': df_to_save.to_parquet(output_file, index=False)
                elif extension == '.csv': df_to_save.to_csv(output_file, index=False, encoding='utf-8-sig')
                elif extension == '.xlsx': df_to_save.to_excel(output_file, index=False)
                logger.info(f"Dataset '{name}' saved to {output_file} ({len(df_to_save):,} rows, {len(df_to_save.columns)} cols)")
            except Exception as e:
                logger.error(f"Failed to save dataset '{name}' to {output_file}: {e}")


def process_dax_news_data_enhanced(
    dax_file: str, news_file: str, base_output_path: str,
    dax_date_col: str, news_date_col: str, news_headline_col: str,
    news_lang_col: Optional[str] = None, # Language column in news_df
    output_format: Literal['granular', 'aggregated', 'hybrid'] = 'hybrid',
    aggregation_method: str = 'mean', # Default if only 'aggregated'
    model_name: str = "ProsusAI/finbert", # Allow choosing model
    use_gpu: bool = True
):
    start_time = datetime.now()
    logger.info(f"Starting DAX-News sentiment processing pipeline with model: {model_name}...")
    
    try:
        analyzer = DAXNewsSentimentAnalyzer(model_name=model_name, use_gpu=use_gpu)
        dax_df, news_df = analyzer.load_and_prepare_data(
            dax_file, news_file, dax_date_col, news_date_col, news_headline_col, news_lang_col
        )
        
        datasets_to_save: Dict[str, Optional[pd.DataFrame]] = {}
        
        if output_format == 'granular':
            granular_df = analyzer.create_granular_dataset(
                dax_df, news_df, dax_date_col, news_date_col, news_headline_col, news_lang_col
            )
            datasets_to_save['granular'] = granular_df
        elif output_format == 'aggregated':
            aggregated_df = analyzer.create_aggregated_dataset(
                dax_df, news_df, dax_date_col, news_date_col, news_headline_col, news_lang_col,
                aggregation_method=aggregation_method, include_headline_details=True
            )
            datasets_to_save[f'aggregated_{aggregation_method}'] = aggregated_df
        elif output_format == 'hybrid':
            granular_df, aggregated_dfs_map = analyzer.create_hybrid_dataset(
                dax_df, news_df, dax_date_col, news_date_col, news_headline_col, news_lang_col
            )
            datasets_to_save['granular'] = granular_df
            for method, df_agg in aggregated_dfs_map.items():
                datasets_to_save[f'aggregated_{method}'] = df_agg
        else:
            raise ValueError(f"Invalid output_format: {output_format}. Choose 'granular', 'aggregated', or 'hybrid'.")

        analyzer.save_datasets(datasets_to_save, base_output_path)
        
        end_time = datetime.now()
        logger.info(f"Pipeline completed in {end_time - start_time}.")
        # ... (summary print remains the same) ...
        print("\n" + "="*80)
        print("ENHANCED DATASET PROCESSING SUMMARY")
        print("="*80)
        print(f"Sentiment Model Used: {model_name}")
        print(f"Output format: {output_format}")
        print(f"Processing time: {end_time - start_time}")
        
        for dataset_name, df_res in datasets_to_save.items():
            if df_res is not None and not df_res.empty:
                print(f"\n{dataset_name.upper()} DATASET:")
                print(f"  Records: {len(df_res):,}")
                
                # Try to find a date column for date range summary
                date_col_to_print = None
                if dax_date_col in df_res.columns and pd.api.types.is_datetime64_any_dtype(df_res[dax_date_col]):
                    date_col_to_print = dax_date_col
                elif not df_res.select_dtypes(include=['datetime64']).empty:
                    date_col_to_print = df_res.select_dtypes(include=['datetime64']).columns[0]

                if date_col_to_print:
                     print(f"  Date range: {df_res[date_col_to_print].min().strftime('%Y-%m-%d')} to {df_res[date_col_to_print].max().strftime('%Y-%m-%d')}")
                
                if 'sentiment_score' in df_res.columns and pd.api.types.is_numeric_dtype(df_res['sentiment_score']):
                    print(f"  Avg daily aggregated sentiment score: {df_res['sentiment_score'].mean():.4f}")
                if 'headline_sentiment_score' in df_res.columns and pd.api.types.is_numeric_dtype(df_res['headline_sentiment_score']):
                    print(f"  Avg individual headline sentiment score: {df_res['headline_sentiment_score'].mean():.4f}")
            else:
                 print(f"\n{dataset_name.upper()} DATASET: Empty or not generated.")
        
        print("="*80)

        return datasets_to_save
        
    except Exception as e:
        logger.error(f"Main processing pipeline failed: {type(e).__name__} - {str(e)}", exc_info=True) # Add exc_info for traceback
        # raise # Optionally re-raise the exception if needed by calling code

# --- ADJUST THESE PARAMETERS FOR YOUR ACTUAL FILES AND DESIRED SETTINGS ---
if __name__ == "__main__":
    # Can be run from anywhere now (all paths are repo-root-anchored, not
    # CWD-relative). If financial_data.py / download_gdelt_gkg.py /
    # process_gdelt_leads.py / prepare_headlines_for_sentiment.py already
    # ran as part of this pipeline invocation, this picks up THEIR exact
    # output files via the shared manifest (data/raw/pipeline_manifest.json)
    # instead of always falling back to the fixed shipped historical
    # dataset — that fallback-only behavior was what made a fresh pipeline
    # run silently ignore whatever new data you'd just downloaded.
    _manifest = load_manifest()
    DAX_INDEX_DATA_FILE = _manifest.get(
        "dax_price_file",
        os.path.join(DATA_DIR, "dax_index_data_20231201_20250528.parquet")  # fallback: shipped historical dataset
    )
    NEWS_HEADLINES_FILE = _manifest.get(
        "prepared_headlines_file",
        os.path.join(DATA_DIR, "raw", "processed_gdelt_leads", "dax40_prepared_headlines_for_sentiment.parquet")
    )
    if _manifest.get("dax_price_file") or _manifest.get("prepared_headlines_file"):
        logger.info(f"Using pipeline manifest inputs: DAX price file = {DAX_INDEX_DATA_FILE}, "
                    f"news file = {NEWS_HEADLINES_FILE} (date range: "
                    f"{_manifest.get('start_date')} to {_manifest.get('end_date')})")
    else:
        logger.info("No pipeline_manifest.json found (or it's incomplete) — falling back to the "
                     "shipped historical dataset. Run financial_data.py and the GDELT stages first "
                     "to process a specific/new date range instead.")

    # Output files are written into <repo_root>/data/ (e.g.
    # data/dax_final_with_sentiment_granular.parquet) so interface.py and
    # tests/test_prediction.py can find them directly — the original script
    # wrote into whatever directory it happened to be launched from.
    OUTPUT_BASE_FILENAME = os.path.join(DATA_DIR, "dax_final_with_sentiment")

    # Column names from your Parquet files
    DAX_DATE_COLUMN_NAME = 'Date' # Or 'date' - Check your DAX index Parquet file carefully. If it's the index, it needs to be reset.
    NEWS_DATE_COLUMN_NAME = 'publication_date'
    NEWS_HEADLINE_COLUMN_NAME = 'headline_text_for_sentiment'
    NEWS_LANGUAGE_COLUMN_NAME = 'language_gdelt' # Crucial for language-specific model handling

    # Sentiment model choice.
    # NOTE (documented, not changed): this intentionally stays
    # nlptown/bert-base-multilingual-uncased-sentiment, NOT ProsusAI/finbert
    # as used by src/training/train_and_predict.py. That mismatch predates
    # this restructuring and matches how the shipped granular dataset was
    # actually produced — switching it here would be a real methodology
    # change, not a reproducibility fix, so it's left as-is and just
    # documented (see README "Known limitations"). If you're regenerating
    # data specifically to retrain the model, consider switching this to
    # "ProsusAI/finbert" for train/inference consistency.
    CHOSEN_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

    try:
        # This will generate, into <repo_root>/data/:
        # - dax_final_with_sentiment_granular.parquet
        # - dax_final_with_sentiment_aggregated_mean.parquet
        # - dax_final_with_sentiment_aggregated_weighted.parquet
        # - dax_final_with_sentiment_aggregated_max_impact.parquet
        with StageTimer("preprocess.py: sentiment scoring + dataset assembly (stage 5/5)"):
            final_datasets = process_dax_news_data_enhanced(
                dax_file=DAX_INDEX_DATA_FILE,
                news_file=NEWS_HEADLINES_FILE,
                base_output_path=OUTPUT_BASE_FILENAME,
                dax_date_col=DAX_DATE_COLUMN_NAME,
                news_date_col=NEWS_DATE_COLUMN_NAME,
                news_headline_col=NEWS_HEADLINE_COLUMN_NAME,
                news_lang_col=NEWS_LANGUAGE_COLUMN_NAME, # Pass the language column
                model_name=CHOSEN_MODEL,                 # Pass the chosen model
                output_format='hybrid',
                # aggregation_method='mean', # Only needed if output_format='aggregated'
                use_gpu=True # Set to False if you don't have a CUDA GPU or MPS
            )

        if final_datasets:
            logger.info("Successfully completed processing and saved datasets.")
            # You can add more printouts or checks for final_datasets here if needed
            # For example, print the head of one of the aggregated dataframes:
            # if 'aggregated_mean' in final_datasets and final_datasets['aggregated_mean'] is not None:
            #     print("\nSample of Aggregated (Mean) Data:")
            #     print(final_datasets['aggregated_mean'].head())
        else:
            logger.warning("Processing finished, but no datasets were returned/saved. Check logs for errors or empty inputs.")

    except FileNotFoundError as fnf_error:
        logger.error(f"A required input file was not found: {fnf_error}")
        logger.error("Please ensure both DAX index data and prepared news headlines Parquet files exist at the specified paths.")
    except ValueError as val_error:
        logger.error(f"A ValueError occurred (often due to incorrect column names or data issues): {val_error}")
    except RuntimeError as rt_error: # Catch PyTorch/CUDA related runtime errors
        logger.error(f"A RuntimeError occurred (often model loading or device issue): {rt_error}")
    except Exception as general_error:
        logger.error(f"An unexpected error occurred during the main execution: {type(general_error).__name__} - {general_error}", exc_info=True)