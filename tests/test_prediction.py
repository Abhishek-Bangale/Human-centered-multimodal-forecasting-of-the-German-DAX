import sys
import os

# This file lives at <repo_root>/tests/test_prediction.py — anchor everything to
# the repo root so it runs correctly regardless of the working directory it's
# launched from, and make src/app importable for prediction_utils.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.append(os.path.join(REPO_ROOT, "src", "app"))

import pandas as pd
import numpy as np
from prediction_utils import DAXModelUtils

def test_prediction_system():
    print("🧪 Testing DAX Prediction System...")

    # Check if we have the required files
    data_file = os.path.join(REPO_ROOT, "data", "dax_final_with_sentiment_granular.parquet")
    model_file = os.path.join(REPO_ROOT, "models", "best_dax_model_final.pth")
    
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return False
    
    if not os.path.exists(model_file):
        print(f"❌ Model file not found: {model_file}")
        return False
    
    print(f"✅ Found data file: {data_file}")
    print(f"✅ Found model file: {model_file}")
    
    try:
        # Initialize the prediction system
        dax_utils = DAXModelUtils(sequence_length=30)
        
        # Load and prepare data
        print("📊 Loading and preparing data...")
        df = dax_utils.load_and_prepare_data(data_file)
        print(f"✅ Loaded {len(df)} records")
        
        # Load the trained model
        print("🤖 Loading trained model...")
        dax_utils.load_model(model_file)
        print("✅ Model loaded successfully")
        
        # Make prediction
        print("🔮 Making prediction...")
        next_price, pred_date, top_headlines, explanation = dax_utils.predict_and_explain(df)
        
        print(f"\n🎯 PREDICTION RESULTS:")
        print(f"📈 Next DAX Price: {next_price:.2f}")
        print(f"📅 Prediction Date: {pred_date.date()}")
        print(f"📰 Top Headlines Found: {len(top_headlines)}")
        
        if not top_headlines.empty:
            print(f"\n📰 TOP IMPACTFUL HEADLINES:")
            for i, (_, row) in enumerate(top_headlines.head(3).iterrows()):
                print(f"{i+1}. {row['headline'][:80]}... (Impact: {row['impact']:.4f})")
        
        print(f"\n💡 EXPLANATION:")
        print(explanation[:500] + "..." if len(explanation) > 500 else explanation)
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_prediction_system()
    if success:
        print("\n🎉 Prediction system test completed successfully!")
    else:
        print("\n💥 Prediction system test failed!") 