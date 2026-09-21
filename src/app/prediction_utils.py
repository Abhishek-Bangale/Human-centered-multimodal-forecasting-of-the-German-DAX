import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import timedelta
import requests
import json
import re
import warnings
warnings.filterwarnings('ignore')

# This file lives at <repo_root>/src/app/prediction_utils.py — used to build a
# default model path that works no matter what directory a caller is run from.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_DEFAULT_MODEL_PATH = os.path.join(_REPO_ROOT, "models", "best_dax_model_final.pth")
_DEFAULT_SCALER_FEATURES_PATH = os.path.join(_REPO_ROOT, "models", "scaler_features.gz")
_DEFAULT_SCALER_TARGET_PATH = os.path.join(_REPO_ROOT, "models", "scaler_target.gz")

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


class DAXModelUtils:
    def __init__(self, sequence_length=30):
        self.sequence_length = sequence_length
        self.scaler_features = MinMaxScaler()
        self.scaler_target = MinMaxScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_columns = None
        self.model = None
        print(f"✅ Using device: {self.device}")

    def load_and_prepare_data(self, path):
        df = pd.read_parquet(path)
        df.columns = [re.match(r"\('([^']*)',", col).group(1) if re.match(r"\('([^']*)',", col) else col for col in df.columns]
        df['Date'] = pd.to_datetime(df['Date'])
        if 'news_date' in df.columns:
            df['news_date'] = pd.to_datetime(df['news_date'])
        df = df.sort_values('Date').reset_index(drop=True)

        # `df` at this point is the *granular* dataset: one row per
        # (trading day, headline) pair, with OHLCV repeated for every headline
        # published that day. The trained model expects one row PER TRADING
        # DAY with daily-aggregated sentiment (news_sentiment_score = mean
        # headline sentiment, news_num_headlines = headline count) — the same
        # aggregation train_and_predict.py performs before training. We
        # reconstruct that here, and keep the original per-headline rows
        # around separately so create_sequences() can still show real
        # headlines/URLs for the explainability panel.
        self._headlines_df = df[['Date', 'headline_text_analyzed', 'source_url']] \
            .dropna(subset=['headline_text_analyzed']).reset_index(drop=True)

        ohlcv_cols = [c for c in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'] if c in df.columns]
        daily_df = df[ohlcv_cols].drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)

        if 'headline_sentiment_score' in df.columns:
            daily_sentiment = df.groupby('Date').agg(
                news_sentiment_score=('headline_sentiment_score', 'mean'),
                news_num_headlines=('headline_text_analyzed', 'count')
            ).reset_index()
            daily_df = daily_df.merge(daily_sentiment, on='Date', how='left')
            daily_df['news_sentiment_score'] = daily_df['news_sentiment_score'].fillna(0.0)
            daily_df['news_num_headlines'] = daily_df['news_num_headlines'].fillna(0)

        daily_df = self.create_features(daily_df)
        return daily_df

    def create_features(self, df):
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

        tech_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Price_Change', 'Price_Return',
                          'High_Low_Ratio', 'Volume_MA', 'SMA_5', 'SMA_10', 'EMA_5', 'RSI', 'MACD', 'Volatility']
        # Match train_and_predict.py exactly: the model was trained on these
        # 15 technical features PLUS 2 daily sentiment features (17 total).
        sentiment_features = []
        if 'news_sentiment_score' in df.columns: sentiment_features.append('news_sentiment_score')
        if 'news_num_headlines' in df.columns: sentiment_features.append('news_num_headlines')
        self.feature_columns = tech_features + sentiment_features

        return df.dropna(subset=self.feature_columns + ['Adj Close']).reset_index(drop=True)

    def calculate_rsi(self, prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_macd(self, prices, fast=12, slow=26):
        return prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()

    def create_sequences(self, df):
        # `df` here is the daily-aggregated dataframe built in
        # load_and_prepare_data — each row/step IS one trading day, matching
        # how the model was trained (30 trading days per sequence, not 30
        # arbitrary headline rows).
        features = df[self.feature_columns].values
        targets = df['Adj Close'].values.reshape(-1,1)
        features_scaled = self.scaler_features.transform(features)
        targets_scaled = self.scaler_target.transform(targets).flatten()
        headlines_df = getattr(self, '_headlines_df', None)
        sequences, target_vals, headlines = [], [], []
        for i in range(self.sequence_length, len(features_scaled)):
            sequences.append(features_scaled[i-self.sequence_length:i])
            target_vals.append(targets_scaled[i])
            if headlines_df is not None and not headlines_df.empty:
                window_start = df['Date'].iloc[i-self.sequence_length]
                window_end = df['Date'].iloc[i-1]
                mask = (headlines_df['Date'] >= window_start) & (headlines_df['Date'] <= window_end)
                headlines.append(headlines_df.loc[mask, ['Date', 'headline_text_analyzed', 'source_url']])
            else:
                headlines.append(pd.DataFrame(columns=['Date', 'headline_text_analyzed', 'source_url']))
        return np.array(sequences), np.array(target_vals), headlines

    def load_model(self, path: str = None):
        if path is None:
            path = _DEFAULT_MODEL_PATH
        """
        Build the network, load its weights, and move everything onto the
        same device (`cpu` on a CPU-only box, otherwise the first CUDA GPU).

        • `map_location=self.device` makes PyTorch rewrite any ‘cuda:0’
          tensors saved in the file so they live on the CPU when CUDA
          is absent.  
        • The optional “module.” block handles checkpoints that were saved
          with `nn.DataParallel` / `DistributedDataParallel`.
        """
        input_size = len(self.feature_columns)
        self.model = DAXLSTMPredictor(input_size)          # stay on CPU for now

        # --- 1. Load the checkpoint, forcing storages onto the current device
        state_dict = torch.load(path, map_location=self.device)

        # --- 2. Strip the `module.` prefix if the model was saved via DataParallel
        if isinstance(state_dict, dict) and next(iter(state_dict)).startswith("module."):
            from collections import OrderedDict
            state_dict = OrderedDict(
                (k.replace("module.", ""), v) for k, v in state_dict.items()
            )

        # --- 3. Restore weights, move the network, set eval-mode
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        print(f"✅ Loaded model from {path} on {self.device}")

        # Load the exact scalers fit during training (previously the code
        # re-fit fresh MinMaxScalers on whatever data was passed in at
        # inference time — see README "Known limitations"). Reusing the
        # shipped fit keeps inference on the same scale the model was
        # trained on.
        self.scaler_features = joblib.load(_DEFAULT_SCALER_FEATURES_PATH)
        self.scaler_target = joblib.load(_DEFAULT_SCALER_TARGET_PATH)
        print(f"✅ Loaded scalers from {os.path.dirname(_DEFAULT_SCALER_FEATURES_PATH)}")


    def predict_and_explain(self, df):
        # NOTE on this method's design (fixed from the original):
        #
        # The original implementation looped over ALL historical 30-day
        # windows (hundreds of them), and for each one iterated row-by-row
        # (`.iterrows()`) over every headline published during that window
        # to accumulate "impact scores" — then, separately, ran the model
        # a second time on the *last* window to get the next-day price.
        #
        # On the real granular dataset that loop is infeasible: each window
        # can contain tens of thousands of headline rows, and summed across
        # ~276 windows that's 15+ million row-iterations, which made the
        # app hang indefinitely (confirmed by profiling during testing).
        #
        # It's also unnecessary: only the most recent window ever drives
        # what's shown to the user (the next-day forecast + its headline
        # explanation), and that window's forward pass is IDENTICAL to the
        # "next day" forward pass the original code ran a second time
        # (verified: sequences[-1] is the same array both places). So this
        # version runs the model once, on the last window only, and scores
        # only that window's headlines — using the Impact Score formula
        # (Prediction Error × |Mean Attention Weight|) applied per-headline,
        # computed with vectorized pandas instead of
        # `.iterrows()` so it stays fast even with tens of thousands of
        # headlines in a single window.
        sequences, targets, headlines_data = self.create_sequences(df)

        with torch.no_grad():
            last_seq_t = torch.FloatTensor(sequences[-1]).unsqueeze(0).to(self.device)
            pred, attn = self.model(last_seq_t)

            pred_val = self.scaler_target.inverse_transform(
                pred.detach().cpu().numpy()
            )[0][0]
            actual_val = self.scaler_target.inverse_transform([[targets[-1]]])[0][0]
            recent_error = abs(pred_val - actual_val)
            mean_attn = attn.mean().item()

        # Same forward pass serves both the "next day" forecast and the
        # error term used in the impact score — no redundant second pass.
        next_price = pred_val
        pred_date = df['Date'].iloc[-1] + timedelta(days=1)

        # Impact Score = Prediction Error × |Mean Attention Weight|,
        # applied to the headlines inside the most recent 30-day window only —
        # the window that actually produced this forecast.
        window_headlines = headlines_data[-1].copy()
        if not window_headlines.empty:
            window_headlines['impact'] = recent_error * abs(mean_attn)
            df_impact = window_headlines.rename(
                columns={'headline_text_analyzed': 'headline', 'source_url': 'url', 'Date': 'date'}
            ).sort_values('impact', ascending=False).drop_duplicates('headline')
        else:
            df_impact = pd.DataFrame(columns=['headline', 'url', 'date', 'impact'])
        top10 = df_impact.head(10)

        prompt = "Explain why these top 10 headlines had the most impact on the DAX prediction:\n"
        for _, row in top10.iterrows():
            prompt += f"- {row['date'].date()}: {row['headline'][:120]}... (Link: {row['url']})\n"
        explanation = self.ask_ollama(prompt)

        return next_price, pred_date, top10, explanation


    def ask_ollama(self, prompt, timeout=15):
        # The original version had no timeout and no error handling: if
        # Ollama isn't installed/running (a real external dependency, not
        # bundled with this repo), `requests.post` would hang or raise and
        # take down the whole app/test run. This fails gracefully instead.
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={"model": "phi4-mini:latest", "prompt": prompt},
                stream=True,
                timeout=timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return (
                "⚠️ Could not reach Ollama at http://localhost:11434 "
                f"({e}). Install Ollama and run `ollama pull phi4-mini` "
                "then `ollama serve` to enable natural-language explanations."
            )

        full_text = ""
        for line in response.iter_lines():
            if line:
                try:
                    obj = json.loads(line.decode())
                    full_text += obj.get("response", "")
                except (json.JSONDecodeError, UnicodeDecodeError):
                    pass
        return full_text
