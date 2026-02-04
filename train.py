
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import config
from src.data.preprocessing import load_and_preprocess_data
from src.data.dataset import create_dataloaders
from src.models.recsys_model import V2ModelResidual
from src.training.trainer import train_model

def main():

    train_df, test_df, num_city, num_country, scaler = load_and_preprocess_data()
    

    config.NUM_CITY = num_city
    config.NUM_COUNTRY = num_country
    

    train_loader, test_loader = create_dataloaders(train_df, test_df, config.BATCH_SIZE)
    

    model = V2ModelResidual(
        city_corpus_size=num_city,
        city_embedding_dim=config.CITY_EMBEDDING_DIM,
        country_corpus_size=num_country,
        country_embedding_dim=config.COUNTRY_EMBEDDING_DIM,
        window_size=config.WINDOW_SIZE,
        hidden_dim=config.HIDDEN_DIM,
        num_residual_blocks=config.NUM_RESIDUAL_BLOCKS,
        num_numeric_features=config.NUM_NUMERIC_FEATURES,
        dropout=config.DROPOUT
    ).to(config.DEVICE)
    
    print(f"Model initialized on {config.DEVICE}")
    

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )
    

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=0)

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        num_epochs=config.NUM_EPOCHS,
        device=config.DEVICE
    )

if __name__ == "__main__":
    main()
