
import torch
import os
import numpy as np
from src.config import config
from src.models.recsys_model import V2ModelResidual, EmbeddingLayer, ResidualBlock

class NextTripPredictor:
    def __init__(self, checkpoint_prefix=None, device=None):
        self.device = device if device else config.DEVICE
        

        if checkpoint_prefix is None:
            checkpoint_prefix = os.path.join(config.CHECKPOINT_DIR, 'best')
            
        self.checkpoint_prefix = checkpoint_prefix
        self.model = self._load_model()
        
    def _load_model(self):
        print(f"Loading model components from prefix: {self.checkpoint_prefix}")

        model = V2ModelResidual(
            city_corpus_size=config.NUM_CITY,
            city_embedding_dim=config.CITY_EMBEDDING_DIM,
            country_corpus_size=config.NUM_COUNTRY,
            country_embedding_dim=config.COUNTRY_EMBEDDING_DIM,
            window_size=config.WINDOW_SIZE,
            hidden_dim=config.HIDDEN_DIM,
            num_residual_blocks=config.NUM_RESIDUAL_BLOCKS,
            num_numeric_features=config.NUM_NUMERIC_FEATURES,
            dropout=config.DROPOUT
        ).to(self.device)
        
        try:

            model.city_embedding.load_state_dict(torch.load(f"{self.checkpoint_prefix}_city_embedding.pt", map_location=self.device))
            model.country_embedding.load_state_dict(torch.load(f"{self.checkpoint_prefix}_country_embedding.pt", map_location=self.device))
            model.input_projection.load_state_dict(torch.load(f"{self.checkpoint_prefix}_input_projection.pt", map_location=self.device))
            model.output_projection.load_state_dict(torch.load(f"{self.checkpoint_prefix}_output_projection.pt", map_location=self.device))
            
            for i, residual_block in enumerate(model.residual_blocks):
                residual_block.load_state_dict(torch.load(f"{self.checkpoint_prefix}_residual_block_{i}.pt", map_location=self.device))
                
            model.eval()
            print("Model loaded successfully.")
            return model
            
        except FileNotFoundError as e:
            print(f"Error loading model: {e}")
            print("Ensure you have trained the model and checkpoints exist.")
            raise

    def predict(self, window_cities, window_countries, numeric_features, k=4):

        w_city_tensor = torch.tensor([window_cities], dtype=torch.long).to(self.device)
        w_country_tensor = torch.tensor([window_countries], dtype=torch.long).to(self.device)
        num_feat_tensor = torch.tensor([numeric_features], dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logits = self.model(w_city_tensor, w_country_tensor, num_feat_tensor)
            _, top_preds = torch.topk(logits, k=k, dim=1)
            
        return top_preds.cpu().numpy()[0]

    def get_embeddings(self, city_id=None, country_id=None):

        with torch.no_grad():
            city_emb = None
            country_emb = None
            
            if city_id is not None:
                city_t = torch.tensor([city_id], dtype=torch.long).to(self.device)
                city_emb = self.model.city_embedding(city_t).cpu().numpy()[0]
                
            if country_id is not None:
                country_t = torch.tensor([country_id], dtype=torch.long).to(self.device)
                country_emb = self.model.country_embedding(country_t).cpu().numpy()[0]
                
            return city_emb, country_emb
