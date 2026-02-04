
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from src.config import config

def load_and_preprocess_data():

    print(f"Loading processed data from {config.TRAIN_FILE} and {config.TEST_FILE}")
    
    try:
        train_df = pd.read_csv(config.TRAIN_FILE).reset_index(drop=True)
        test_df = pd.read_csv(config.TEST_FILE).reset_index(drop=True)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please run 'src/data/feature_engineering.py' first.")
        raise


    city_cols = [f'city_{i}' for i in range(1, config.WINDOW_SIZE + 1)]
    country_cols = [f'country_{i}' for i in range(1, config.WINDOW_SIZE + 1)]
    target_col = 'target_city'


    max_city_train = train_df[city_cols + [target_col]].max().max()
    max_country_train = train_df[country_cols].max().max()


    max_city_test = test_df[city_cols].max().max()
    if target_col in test_df.columns:
         max_city_test = max(max_city_test, test_df[target_col].max())

    max_country_test = test_df[country_cols].max().max()

    num_city = int(max(max_city_train, max_city_test))
    num_country = int(max(max_country_train, max_country_test))

    print(f"Dynamic Max IDs: num_city={num_city}, num_country={num_country}")
    

    config.NUM_CITY = num_city
    config.NUM_COUNTRY = num_country


    numeric_cols = ['avg_stay_duration', 'unique_countries_count', 'avg_gap_duration', 'total_days_so_far']
    scaler = StandardScaler()

    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    print("Numeric columns scaled.")
    
    return train_df, test_df, num_city, num_country, scaler
