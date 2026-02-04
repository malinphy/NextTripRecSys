import pandas as pd

def get_precision_at_k(df, k=4, target_col='target_city'):

    pred_cols = [f'p{i}' for i in range(1, k + 1)]

    for col in pred_cols + [target_col]:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in dataframe.")
            return None


    hits = df[pred_cols].eq(df[target_col], axis=0).any(axis=1)

    precision = hits.mean()
    return precision
