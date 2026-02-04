
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):

    def __init__(self, dim, dropout=0.3):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)

        out = out + residual
        out = F.relu(out)
        return out


class EmbeddingLayer(nn.Module):

    def __init__(self, corpus_size, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(
            corpus_size + 1,
            embedding_dim,
            padding_idx=0
        )

        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)
    
    def forward(self, x):
        return self.embedding(x)


class V2ModelResidual(nn.Module):
    def __init__(self,
                 city_corpus_size,
                 city_embedding_dim,
                 country_corpus_size,
                 country_embedding_dim,
                 window_size=5,
                 hidden_dim=512,
                 num_residual_blocks=3,
                 num_numeric_features=4,
                 dropout=0.3
                 ):
        super(V2ModelResidual, self).__init__()
        
        self.city_embedding = EmbeddingLayer(city_corpus_size, city_embedding_dim)
        self.country_embedding = EmbeddingLayer(country_corpus_size, country_embedding_dim)
        

        linear_input_dim = (window_size * city_embedding_dim) + \
                          (window_size * country_embedding_dim) + \
                          num_numeric_features
        
        self.input_projection = nn.Sequential(
            nn.Linear(linear_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout)
            for _ in range(num_residual_blocks)
        ])
        

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, city_corpus_size + 1)
        )
    
    def forward(self, window_city, window_country, numeric_features):

        city_emb = self.city_embedding(window_city)       
        country_emb = self.country_embedding(window_country)
        

        city_emb = city_emb.view(city_emb.size(0), -1)
        country_emb = country_emb.view(country_emb.size(0), -1)

        x = torch.cat((city_emb, country_emb, numeric_features), dim=1)

        x = self.input_projection(x)

        for residual_block in self.residual_blocks:
            x = residual_block(x)
        

        logits = self.output_projection(x)
        return logits
