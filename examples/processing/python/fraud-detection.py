import pandas as pd
import numpy as np

def load_fraud_data(file_paths):
    try:
        dfs = []
        for file_path in file_paths:
            df = pd.read_csv(file_path)
            dfs.append(df)
        data = pd.concat(dfs, ignore_index=True)
        print(f"Loaded {len(data)} transaction samples from {len(file_paths)} files")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def process_fraud_data(df, test_ratio=0.2, random_seed=42):
    y = df["Class"].values.astype(float)

    # Build feature matrix (V1-V28 + Amount)
    feature_cols = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    X = df[feature_cols].astype(float)

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
    y_train = y[train_idx].reshape(-1, 1)
    X_test = X_normalized.iloc[test_idx].values
    y_test = y[test_idx].reshape(-1, 1)

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    file_paths = [
        "examples/datasets/credit_card_fraud/credit_card_1.csv",
        "examples/datasets/credit_card_fraud/credit_card_2.csv",
        "examples/datasets/credit_card_fraud/credit_card_3.csv",
        "examples/datasets/credit_card_fraud/credit_card_4.csv",
    ]

    df = load_fraud_data(file_paths)

    if df is not None:
        X_train, y_train, X_test, y_test = process_fraud_data(df)

        print("-" * 60)
        print("Dataset Split:")
        print("-" * 60)
        print(f"    Training samples: {X_train.shape[0]}")
        print(f"    Test samples:     {X_test.shape[0]}")
        print(f"    Features:         {X_train.shape[1]}")
        print("-" * 60)

        fraud_train = y_train.sum()
        fraud_test = y_test.sum()
        print(f"Class Distribution:")
        print(f"    Training - Fraud: {int(fraud_train)}, Legitimate: {len(y_train) - int(fraud_train)}")
        print(f"    Test     - Fraud: {int(fraud_test)}, Legitimate: {len(y_test) - int(fraud_test)}")
