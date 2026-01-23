import pandas as pd
import numpy as np

def load_cancer_data(file_path):
    try:
        column_names = ["class", "age", "menopause", "tumor_size", "inv_nodes",
                        "node_caps", "deg_malig", "breast", "breast_quad", "irradiat"]
        data = pd.read_csv(file_path, header=None, names=column_names)
        print(f"Loaded {len(data)} breast cancer samples from {file_path}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def parse_age(val):
    age_map = {"10-19": 0, "20-29": 1, "30-39": 2, "40-49": 3, "50-59": 4,
               "60-69": 5, "70-79": 6, "80-89": 7, "90-99": 8}
    return age_map.get(val, 4)

def parse_menopause(val):
    meno_map = {"lt40": 0, "premeno": 1, "ge40": 2}
    return meno_map.get(val, 1)

def parse_tumor_size(val):
    if pd.isna(val) or val == "?":
        return 5
    try:
        start = int(val.split("-")[0])
        return start // 5
    except:
        return 5

def parse_inv_nodes(val):
    if pd.isna(val) or val == "?":
        return 0
    try:
        start = int(val.split("-")[0])
        return start // 3
    except:
        return 0

def parse_breast_quad(val):
    quad_map = {"left_up": 0, "left_low": 1, "right_up": 2, "right_low": 3, "central": 4}
    return quad_map.get(val, 2)

def process_cancer_data(df, test_ratio=0.2, random_seed=42):
    y = df["class"].apply(lambda x: 1.0 if x == "recurrence-events" else 0.0)

    # Build feature matrix
    X = pd.DataFrame()
    X["age"] = df["age"].apply(parse_age)
    X["menopause"] = df["menopause"].apply(parse_menopause)
    X["tumor_size"] = df["tumor_size"].apply(parse_tumor_size)
    X["inv_nodes"] = df["inv_nodes"].apply(parse_inv_nodes)
    X["node_caps"] = df["node_caps"].apply(lambda x: 1.0 if x == "yes" else 0.0)
    X["deg_malig"] = pd.to_numeric(df["deg_malig"], errors="coerce").fillna(2)
    X["breast"] = df["breast"].apply(lambda x: 1.0 if x == "right" else 0.0)
    X["breast_quad"] = df["breast_quad"].apply(parse_breast_quad)
    X["irradiat"] = df["irradiat"].apply(lambda x: 1.0 if x == "yes" else 0.0)

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
    file_path = "examples/datasets/breast_cancer/breast-cancer.data"

    df = load_cancer_data(file_path)

    if df is not None:
        X_train, y_train, X_test, y_test = process_cancer_data(df)

        print("-" * 60)
        print("Dataset Split:")
        print("-" * 60)
        print(f"    Training samples: {X_train.shape[0]}")
        print(f"    Test samples:     {X_test.shape[0]}")
        print(f"    Features:         {X_train.shape[1]}")
        print("-" * 60)

        recurrence_train = y_train.sum()
        recurrence_test = y_test.sum()
        print(f"Class Distribution:")
        print(f"    Training - Recurrence: {int(recurrence_train)}, No Recurrence: {len(y_train) - int(recurrence_train)}")
        print(f"    Test     - Recurrence: {int(recurrence_test)}, No Recurrence: {len(y_test) - int(recurrence_test)}")
