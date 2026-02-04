
import torch
import torch.nn as nn
import time
import os
import numpy as np
import pandas as pd
from src.config import config

def save_model_components(model, path_prefix):

    os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
    
    torch.save(model.city_embedding.state_dict(), f"{path_prefix}_city_embedding.pt")
    torch.save(model.country_embedding.state_dict(), f"{path_prefix}_country_embedding.pt")
    torch.save(model.input_projection.state_dict(), f"{path_prefix}_input_projection.pt")
    torch.save(model.output_projection.state_dict(), f"{path_prefix}_output_projection.pt")
    

    for i, residual_block in enumerate(model.residual_blocks):
        torch.save(residual_block.state_dict(), f"{path_prefix}_residual_block_{i}.pt")

def save_embeddings_only(model, path_prefix):

    torch.save(model.city_embedding.state_dict(), f"{path_prefix}_city_embedding.pt")
    torch.save(model.country_embedding.state_dict(), f"{path_prefix}_country_embedding.pt")

def save_residual_blocks_only(model, path_prefix):

    for i, residual_block in enumerate(model.residual_blocks):
        torch.save(residual_block.state_dict(), f"{path_prefix}_residual_block_{i}.pt")

def get_precision_at_k_df(df, k=4, target_col='target_city'):

    pred_cols = [f'p{i}' for i in range(1, k + 1)]
    hits = df[pred_cols].eq(df[target_col], axis=0).any(axis=1)
    return hits.mean()

def train_model(model, train_loader, test_loader, optimizer, scheduler, criterion, num_epochs, device):
    print("Starting training...")
    
    best_precision = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):

        model.train()
        running_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
 
            window_city = batch['window_city'].to(device) 
            window_country = batch['window_country'].to(device)
            numeric_features = batch['numeric_features'].to(device)
            target_city = batch['target_city'].to(device)
            
            optimizer.zero_grad()
            

            logits = model(window_city, window_country, numeric_features)
            

            loss = criterion(logits, target_city)
            loss.backward()
            

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch [{batch_idx+1}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        

        current_prefix = f"{config.PATH_PREFIX}_epoch_{epoch+1}"

        model.eval()
        all_preds = []
        all_targets = []
        val_loss = 0.0
        
        print("Evaluating on test dataset...")
        with torch.no_grad():
            for batch in test_loader:
                window_city = batch['window_city'].to(device)
                window_country = batch['window_country'].to(device)
                numeric_features = batch['numeric_features'].to(device)
                target = batch['target_city'].to(device)
                
                outputs = model(window_city, window_country, numeric_features)
                loss = criterion(outputs, target)
                val_loss += loss.item()
                
                # Top 4 preds
                _, top_preds = torch.topk(outputs, k=4, dim=1)
                
                all_preds.append(top_preds.cpu())
                all_targets.append(target.cpu())
        
        val_loss /= len(test_loader)
        all_preds = torch.cat(all_preds, dim=0).numpy()
        all_targets = torch.cat(all_targets, dim=0).numpy()
        
        # Calculate Precision@K
        data = np.hstack([all_targets.reshape(-1, 1), all_preds])
        cols = ['target_city'] + [f'p{i+1}' for i in range(4)]
        prediction_df = pd.DataFrame(data, columns=cols)
        pres = get_precision_at_k_df(prediction_df, k=4)
        

        print("=" * 70)
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  Precision@4: {pres:.4f}")
        

        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        if old_lr != new_lr:
            print(f"  Learning rate reduced: {old_lr:.6f} -> {new_lr:.6f}")
        

        if pres > best_precision:
            best_precision = pres
            best_epoch = epoch + 1
            

            best_model_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'precision': pres,
                'loss': val_loss,
            }, best_model_path)
            

            best_component_prefix = os.path.join(config.CHECKPOINT_DIR, 'best')
            save_model_components(model, best_component_prefix)
            
            print(f"  ✓ New best model saved! (Precision@4: {pres:.4f})")
        
        print("=" * 70)
        print()
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print(f"Best Precision@4: {best_precision:.4f} (Epoch {best_epoch})")
    print("=" * 70)
