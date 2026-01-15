import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, accuracy_score

def train_and_save():
    print("--- Starting Model Training Pipeline ---")
    try:
        # Load dataset
        df = pd.read_csv('yahooStock.csv')
        
        # 1. Data Cleaning & Label Encoding
        df['transactionType_cleaned'] = df['transactionType'].astype(str).str.strip().str.upper()
        df_filtered = df[df['transactionType_cleaned'].isin(['BUY', 'SELL'])].copy()
        df_filtered['Target'] = np.where(df_filtered['transactionType_cleaned'] == 'BUY', 1, 0)

        # 2. Feature Selection
        numerical_features = ['amount', 'reportedPrice', 'usdValue']
        for col in numerical_features:
            df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')
        
        df_final = df_filtered.dropna(subset=numerical_features + ['Target'])
        X = df_final[numerical_features]
        y = df_final['Target']

        # 3. Data Split & Scaling
        # Using a 75/25 split as per standard ML practice
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)
        
        # 4. Addressing Imbalance (SMOTE)
        smote = SMOTE(random_state=42)
        x_res, y_res = smote.fit_resample(x_train_scaled, y_train)

        # 5. Model Training (Extra Trees Classifier)
        print("Training Extra Trees Classifier (The Best Model)...")
        model = ExtraTreesClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        model.fit(x_res, y_res)

        # 6. Evaluation
        predictions = model.predict(x_test_scaled)
        print(f"Validation Accuracy: {accuracy_score(y_test, predictions):.4f}")
        
        # 7. Save Artifacts for Deployment
        joblib.dump(model, 'best_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        print("Done! 'best_model.pkl' and 'scaler.pkl' have been generated.")

    except Exception as e:
        print(f"Error occurred: {e}")

if __name__ == "__main__":
    train_and_save()