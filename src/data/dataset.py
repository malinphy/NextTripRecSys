
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from src.config import config

class BookingDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.numeric_cols = ['avg_stay_duration', 'unique_countries_count', 'avg_gap_duration', 'total_days_so_far']
        self.window_size = config.WINDOW_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        

        city_cols = [f'city_{i}' for i in range(1, self.window_size + 1)]
        country_cols = [f'country_{i}' for i in range(1, self.window_size + 1)]

        window_city = row[city_cols].values.tolist()
        window_country = row[country_cols].values.tolist()
        
        target_city = row.get('target_city', 0) 

        numeric_features = torch.tensor(row[self.numeric_cols].values.astype(np.float32))

        item = {
            'window_city': torch.tensor(window_city, dtype=torch.long),     
            'window_country': torch.tensor(window_country, dtype=torch.long),
            'numeric_features': numeric_features,
            'target_city': torch.tensor(target_city, dtype=torch.long)
        }

        return item

def create_dataloaders(train_df, test_df, batch_size):

    
    train_dataset = BookingDataset(train_df)
    test_dataset = BookingDataset(test_df)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader
