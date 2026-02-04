import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

def load_data(data_path):
    
    train = pd.read_csv(os.path.join(data_path, 'train_set.csv'))
    test = pd.read_csv(os.path.join(data_path, 'test_set.csv'))
    gt = pd.read_csv(os.path.join(data_path, 'ground_truth.csv'))
    

    for df in [train, test]:
        df['checkin'] = pd.to_datetime(df['checkin'])
        df['checkout'] = pd.to_datetime(df['checkout'])
    
    return train, test, gt

def prepare_test_with_gt(test_df, gt_df):

    test_complete = test_df.copy()
    

    gt_map_city = gt_df.set_index('utrip_id')['city_id'].to_dict()
    gt_map_country = gt_df.set_index('utrip_id')['hotel_country'].to_dict()
    
    mask = test_complete['city_id'] == 0
    test_complete.loc[mask, 'city_id'] = test_complete.loc[mask, 'utrip_id'].map(gt_map_city)
    test_complete.loc[mask, 'hotel_country'] = test_complete.loc[mask, 'utrip_id'].map(gt_map_country)
    
    return test_complete

def fit_label_encoders(dfs, cols):

    encoders = {}
    for col in cols:
        le = LabelEncoder()

        all_vals = pd.concat([df[col].astype(str) for df in dfs]).unique()
        le.fit(all_vals)
        encoders[col] = le
        print(f"{col} icin {len(le.classes_)} sinif bulundu.")
    return encoders

def transform_labels(df, encoders):
    df_encoded = df.copy()
    for col, le in encoders.items():
        df_encoded[col] = le.transform(df[col].astype(str)) + 1
    return df_encoded

def add_features(df):

    df = df.sort_values(['utrip_id', 'checkin']).reset_index(drop=True)
    
    df['pos'] = df.groupby('utrip_id').cumcount() + 1
    
    df['stay_duration'] = (df['checkout'] - df['checkin']).dt.days
    
    df['prev_checkout'] = df.groupby('utrip_id')['checkout'].shift(1)
    df['gap_duration'] = (df['checkin'] - df['prev_checkout']).dt.days.fillna(0)
    
    df['first_checkin'] = df.groupby('utrip_id')['checkin'].transform('min')
    df['days_from_start'] = (df['checkin'] - df['first_checkin']).dt.days
    
    df['checkin_dow'] = df['checkin'].dt.dayofweek
    df['checkin_month'] = df['checkin'].dt.month
    
    df['prev_country'] = df.groupby('utrip_id')['hotel_country'].shift(1)
    df['country_change'] = (df['hotel_country'] != df['prev_country']).astype(int)
    
    df['is_revisit'] = df.groupby(['utrip_id', 'city_id']).cumcount() > 0
    df['is_revisit'] = df['is_revisit'].astype(int)
    
    return df

def create_windows(df, window_size=5):
    
    data = []
    
    grouped = df.groupby('utrip_id')
    
    for utrip_id, group in grouped:
        cities = group['city_id'].tolist()
        countries = group['hotel_country'].tolist()
        stays = group['stay_duration'].tolist()
        gaps = group['gap_duration'].tolist()
        days_from_start = group['days_from_start'].tolist()
        
        if len(cities) < 2: continue
        
        for i in range(len(cities) - 1):
            target = cities[i+1]
            

            start_idx = max(0, i - window_size + 1)
            

            win_cities = cities[start_idx : i+1]
            win_countries = countries[start_idx : i+1]
            win_stays = stays[start_idx : i+1]
            win_gaps = gaps[start_idx : i+1]

            avg_stay = np.mean(win_stays)
            unique_countries = len(set(win_countries))
            avg_gap = np.mean(win_gaps)
            total_days_spent = days_from_start[i] + stays[i]
            

            pad_len = window_size - len(win_cities)
            win_cities = [0]*pad_len + win_cities
            win_countries = [0]*pad_len + win_countries
            
            win_dict = {
                'utrip_id': utrip_id, 
                'target_city': target,
                'avg_stay_duration': avg_stay,
                'unique_countries_count': unique_countries,
                'avg_gap_duration': avg_gap,
                'total_days_so_far': total_days_spent
            }
            

            for j in range(window_size):
                win_dict[f'city_{j}'] = win_cities[j]
                win_dict[f'country_{j}'] = win_countries[j]
            
            data.append(win_dict)
            
    return pd.DataFrame(data)

def main():
    PATH = 'e:/PERSONAL_PROJ/recsys/data/'
    
    train_raw, test_raw, gt_raw = load_data(PATH)
    

    test_complete = prepare_test_with_gt(test_raw, gt_raw)
    

    encoders = fit_label_encoders([train_raw, test_complete], ['city_id', 'hotel_country', 'device_class', 'booker_country', 'affiliate_id'])
    
    train_enc = transform_labels(train_raw, encoders)
    test_enc = transform_labels(test_complete, encoders)
    

    train_fe = add_features(train_enc)
    test_fe = add_features(test_enc)


    train_windows = create_windows(train_fe)
    test_windows_all = create_windows(test_fe)

    final_test = test_windows_all.groupby('utrip_id').tail(1)
    test_intermediate = test_windows_all.drop(final_test.index)
    

    train_final = pd.concat([train_windows, test_intermediate]).reset_index(drop=True)
    

    print(f"Final Train Shape: {train_final.shape}")
    print(f"Final Test Shape: {final_test.shape}")
    
    train_final.to_csv(os.path.join(PATH, 'train_fe_eng.csv'), index=False)
    final_test.to_csv(os.path.join(PATH, 'final_test.csv'), index=False)


if __name__ == "__main__":
    main()
