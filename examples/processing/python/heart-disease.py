import pandas as pd
import numpy as np

def load_heart_data(file_path):
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded {len(data)} heart disease samples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

def process_heart_data(df, test_ratio=0.2, random_seed=42):
    feature_cols = ["male", "age", "currentSmoker",
                    "cigsPerDay", "BPMeds", "prevalentStroke", "prevalentHyp",
                    "diabetes", "totChol", "sysBP", "diaBP",
                    "BMI", "heartRate", "glucose"]

    
    # Extract Features and Target
    X = df[feature_cols].copy()
    y = df['TenYearCHD'].copy()


    binary_cols = ["male", "currentSmoker", "BPMeds", "prevalentStroke", "prevalentHyp",
                   "diabetes"]
    for col in binary_cols:
        X[col] = X[col].apply(lambda x: 1.0 if str(x).lower() == 'yes' else 0.0)


    X = X.astype(float)

    # Normalize Features
    means = X.mean()
    stds = X.std(ddof=0)
    stds[stds < 1e-9] = 1.0
    X_normalized = (X - means) / stds

    # Split Data
    np.random.seed(random_seed)
    indices = np.random.permutation(len(df))
    
    test_size = int(len(df) * test_ratio)
    train_size = len(df) - test_size
    
    train_idx = indices[:train_size]
    test_idx = indices[train_size:]
    
    X_train = X_normalized.iloc[train_idx].values
    y_train = y.iloc[train_idx].values.reshape(-1, 1)
    X_test = X_normalized.iloc[test_idx].values
    y_test = y.iloc[test_idx].values.reshape(-1, 1)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    file_path = "examples/datasets/heart-disease.csv"
    
    df = load_heart_data(file_path)
    
    if df is not None:
        X_train, y_train, X_test, y_test = process_heart_data(df)

        print("-" * 60)
        print("Dataset Split:")
        print("-" * 60)
        print(f"    Training samples: {X_train.shape[0]}")
        print(f"    Test samples:     {X_test.shape[0]}")
        print(f"    Features:         {X_train.shape[1]}")
        print("-" * 60)
        
        print(f"Top 10 Train Y: \n{y_train[:10]}")