import os
import warnings
import traceback # FIXED: Import the traceback module for detailed error printing

# Suppress TensorFlow warnings and oneDNN messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta
import requests
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# This file lives at <repo_root>/src/training/train_and_predict.py — anchor all
# data/model paths to the repo root so the script runs correctly regardless of
# the working directory it's launched from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
MODELS_DIR = os.path.join(REPO_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Deep Learning Model Classes ---
class DAXDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        return {
            'sequence': torch.FloatTensor(self.sequences[idx]),
            'target': torch.FloatTensor([self.targets[idx]])
        }

class DAXLSTMPredictor(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=4, dropout=0.12):
        super(DAXLSTMPredictor, self).__init__()
        actual_heads = min(8, hidden_size)
        while hidden_size % actual_heads != 0 and actual_heads > 1:
            actual_heads -= 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=actual_heads, dropout=dropout, batch_first=True)
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        out = attn_out[:, -1, :]
        out = self.fc(out)
        return out, attn_weights


# --- Main Utility Class with Integrated Sentiment Analysis ---
class DAXModelUtils:
    def __init__(self, sequence_length=30, sentiment_model_name="ProsusAI/finbert"):
        self.sequence_length = sequence_length
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.feature_columns = None
        self.model = None

        self.sentiment_model_name = sentiment_model_name
        self.sentiment_tokenizer = None
        self.sentiment_model = None
        self.label_mapping = {0: 'positive', 1: 'negative', 2: 'neutral'}

        print(f"✅ Using device: {self.device}")
        self._load_sentiment_model()

    def _load_sentiment_model(self):
        print(f"Loading sentiment model: {self.sentiment_model_name}")
        try:
            self.sentiment_tokenizer = AutoTokenizer.from_pretrained(self.sentiment_model_name)
            self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                self.sentiment_model_name,
                use_safetensors=True
            )
            self.sentiment_model.to(self.device)
            self.sentiment_model.eval()
            print("✅ Sentiment model loaded successfully (via safetensors).")
        except Exception as e:
            print(f"❌ Error loading sentiment model: {e}")
            raise

    # CHANGED: Reduced the default batch size from 32 to 16 to prevent GPU out-of-memory errors.
    def analyze_sentiment_batch(self, headlines: List[str], batch_size: int = 16) -> List[Dict[str, float]]:
        all_scores = []
        if not headlines:
            return all_scores
            
        for i in tqdm(range(0, len(headlines), batch_size), desc="Analyzing Sentiment"):
            batch = headlines[i:i+batch_size]
            inputs = self.sentiment_tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
            
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            for prediction in predictions:
                probs = prediction.cpu().numpy()
                scores = {self.label_mapping[j]: float(prob) for j, prob in enumerate(probs)}
                all_scores.append(scores)
        return all_scores

    def load_and_prepare_data(self, dax_data_path: str, news_data_path: str):
        print("--- Starting Data Loading and Preparation ---")
        dax_df = pd.read_parquet(dax_data_path)
        if isinstance(dax_df.columns, pd.MultiIndex):
            dax_df.columns = dax_df.columns.get_level_values(0)
        if 'Date' not in dax_df.columns and dax_df.index.name == 'Date':
             dax_df = dax_df.reset_index()
        dax_df['Date'] = pd.to_datetime(dax_df['Date'])
        dax_df = dax_df.sort_values('Date').reset_index(drop=True)
        print(f"Loaded {len(dax_df)} DAX records.")

        news_df = pd.read_parquet(news_data_path)
        news_df['publication_date'] = pd.to_datetime(news_df['publication_date'])
        print(f"Loaded {len(news_df)} news headlines.")

        headlines_to_analyze_df = news_df[news_df['headline_text_for_sentiment'].notna()].copy()
        print(f"Found {len(headlines_to_analyze_df)} non-empty headlines to analyze.")

        if headlines_to_analyze_df.empty:
            print("❌ No valid headlines found to analyze. Proceeding with only technical features.")
            return self.create_features(dax_df)

        headlines_list = headlines_to_analyze_df['headline_text_for_sentiment'].tolist()
        sentiment_scores_list = self.analyze_sentiment_batch(headlines_list)
        
        sentiment_df = pd.DataFrame(sentiment_scores_list)
        headlines_with_sentiment_df = pd.concat([headlines_to_analyze_df.reset_index(drop=True), sentiment_df], axis=1)

        print("Aggregating daily sentiment scores...")
        headlines_with_sentiment_df['date_only'] = headlines_with_sentiment_df['publication_date'].dt.date
        
        daily_sentiment_agg = headlines_with_sentiment_df.groupby('date_only').agg(
            news_sentiment_score=pd.NamedAgg(column='positive', aggfunc='mean'),
            news_sentiment_negative=pd.NamedAgg(column='negative', aggfunc='mean'),
            news_num_headlines=pd.NamedAgg(column='headline_text_for_sentiment', aggfunc='count')
        ).reset_index()
        
        daily_sentiment_agg['news_sentiment_score'] = daily_sentiment_agg['news_sentiment_score'] - daily_sentiment_agg['news_sentiment_negative']
        daily_sentiment_agg.drop(columns=['news_sentiment_negative'], inplace=True)
        
        daily_sentiment_agg.rename(columns={'date_only': 'Date'}, inplace=True)
        daily_sentiment_agg['Date'] = pd.to_datetime(daily_sentiment_agg['Date'])
        
        print("Daily sentiment aggregation complete.")

        print("Merging DAX data with aggregated sentiment...")
        daily_sentiment_agg['merge_date'] = daily_sentiment_agg['Date'] + timedelta(days=1)
        
        merged_df = pd.merge(dax_df, daily_sentiment_agg[['merge_date', 'news_sentiment_score', 'news_num_headlines']], 
                             left_on='Date', right_on='merge_date', how='left')
        
        merged_df[['news_sentiment_score', 'news_num_headlines']] = merged_df[['news_sentiment_score', 'news_num_headlines']].fillna(method='ffill')
        merged_df['news_sentiment_score'].fillna(0, inplace=True)
        merged_df['news_num_headlines'].fillna(0, inplace=True)

        merged_df.drop(columns=['merge_date'], inplace=True)
        print("Merge complete.")

        print("Creating final features for the model...")
        df_final = self.create_features(merged_df)
        return df_final

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df['Price_Change'] = df['Close'] - df['Open']
        df['Price_Return'] = df['Close'].pct_change()
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Volume_MA'] = df['Volume'].rolling(5).mean()
        df['SMA_5'] = df['Close'].rolling(5).mean()
        df['SMA_10'] = df['Close'].rolling(10).mean()
        df['EMA_5'] = df['Close'].ewm(span=5).mean()
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['Volatility'] = df['Price_Return'].rolling(5).std()

        tech_features = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Price_Return',
            'High_Low_Ratio', 'Volume_MA', 'SMA_5', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'Volatility'
        ]
        
        sentiment_features = []
        if 'news_sentiment_score' in df.columns: sentiment_features.append('news_sentiment_score')
        if 'news_num_headlines' in df.columns: sentiment_features.append('news_num_headlines')
        
        self.feature_columns = tech_features + sentiment_features
        
        print(f"Using {len(self.feature_columns)} features for the model: {self.feature_columns}")
        
        return df.dropna(subset=self.feature_columns + ['Adj Close'])

    def calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain.fillna(0) / loss.replace(0, 1e-9)
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26) -> pd.Series:
        return prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[pd.DataFrame]]:
        numeric_features_df = df[self.feature_columns].select_dtypes(include=np.number)
        features = numeric_features_df.values
        targets = df['Adj Close'].values.reshape(-1, 1)

        self.scaler_features.fit(features)
        self.scaler_target.fit(targets)
        
        features_scaled = self.scaler_features.transform(features)
        targets_scaled = self.scaler_target.transform(targets).flatten()

        sequences, target_vals = [], []
        for i in range(self.sequence_length, len(features_scaled)):
            sequences.append(features_scaled[i-self.sequence_length:i])
            target_vals.append(targets_scaled[i])
            
        return np.array(sequences), np.array(target_vals), []


def train_and_save_model():
    """Main function to orchestrate the model training pipeline."""
    print("--- Starting DAX LSTM Model Training Pipeline ---")
    try:
        model_utils = DAXModelUtils(sequence_length=30, sentiment_model_name="ProsusAI/finbert")
    except Exception as e:
        print(f"❌ Initialization failed:")
        traceback.print_exc() # FIXED: Print full traceback on initialization error
        return

    dax_data_file = os.path.join(DATA_DIR, "dax_index_data_20231201_20250528.parquet")
    # NOTE: this intermediate file is produced by running the preprocessing
    # pipeline (src/preprocessing/) end to end — it is NOT shipped in the repo
    # because of its size. Run `python src/preprocessing/get_input_for_prediction.py`
    # first if this file doesn't exist yet.
    news_data_file = os.path.join(DATA_DIR, "raw", "processed_gdelt_leads", "dax40_prepared_headlines_for_sentiment.parquet")

    print("\nStep 1: Loading, processing, and merging data...")
    try:
        final_df = model_utils.load_and_prepare_data(dax_data_file, news_data_file)
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    except Exception as e:
        # FIXED: This block now prints the full error traceback, not just "3".
        print(f"❌ An error occurred during data preparation. See details below:")
        traceback.print_exc()
        return
    
    print("\nStep 2: Creating sequences for training and validation...")
    if len(final_df) < model_utils.sequence_length + 1:
        print("❌ Error: Not enough data in the final DataFrame to create even one training sequence.")
        return
        
    sequences, targets, _ = model_utils.create_sequences(final_df)
    dataset = DAXDataset(sequences, targets)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    print(f"✅ Created {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")
    
    input_size = len(model_utils.feature_columns)
    model = DAXLSTMPredictor(input_size=input_size).to(model_utils.device)
    criterion = nn.MSELoss()
    # FIXED: Changed 'optim.Adam' to 'Adam' to match the import statement.
    optimizer = Adam(model.parameters(), lr=0.0005) 
    
    num_epochs = 50
    best_val_loss = float('inf')
    
    print(f"\n--- Step 3: Starting training for {num_epochs} epochs ---")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training"):
            seq = batch['sequence'].to(model_utils.device)
            target = batch['target'].to(model_utils.device)
            optimizer.zero_grad()
            output, _ = model(seq)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                seq = batch['sequence'].to(model_utils.device)
                target = batch['target'].to(model_utils.device)
                output, _ = model(seq)
                val_loss += criterion(output, target).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_dax_model_final.pth'))
            print(f"✅ Model saved to '{os.path.join(MODELS_DIR, 'best_dax_model_final.pth')}' (Validation loss improved to {avg_val_loss:.6f})")

    print("\n--- Training Finished ---")
    print(f"🏆 Best validation loss achieved: {best_val_loss:.6f}")

if __name__ == '__main__':
    train_and_save_model()
    print("\n--- Saving data scalers for prediction ---")
    
    model_utils = DAXModelUtils(sequence_length=30)
    dax_data_file = os.path.join(DATA_DIR, "dax_index_data_20231201_20250528.parquet")
    news_data_file = os.path.join(DATA_DIR, "raw", "processed_gdelt_leads", "dax40_prepared_headlines_for_sentiment.parquet")
    final_df = model_utils.load_and_prepare_data(dax_data_file, news_data_file)
    model_utils.create_sequences(final_df) # This fits the scalers

    import joblib
    joblib.dump(model_utils.scaler_features, os.path.join(MODELS_DIR, 'scaler_features.gz'))
    joblib.dump(model_utils.scaler_target, os.path.join(MODELS_DIR, 'scaler_target.gz'))
    print(f"✅ Scalers saved to {MODELS_DIR}")