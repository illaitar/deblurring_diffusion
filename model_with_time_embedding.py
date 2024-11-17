
import math
import torch
from torch import nn

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, embedding_dim):
        super(SinusoidalPositionEmbeddings, self).__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = torch.exp(-torch.arange(half_dim, device=timesteps.device) * math.log(10000) / (half_dim - 1))
        emb = timesteps[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_features):
        super(DenseBlock, self).__init__()
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Conv2d(num_features, num_features, 3, 1, 1))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.block(x)


class RRDB(nn.Module):
    def __init__(self, num_features, num_layers=3, scaling_factor=0.2):
        super(RRDB, self).__init__()
        self.dense_blocks = nn.Sequential(
            DenseBlock(num_layers, num_features),
            DenseBlock(num_layers, num_features),
            DenseBlock(num_layers, num_features),
        )
        self.scaling_factor = scaling_factor

    def forward(self, x):
        out = x
        out = out + self.scaling_factor * self.dense_blocks(out)
        return out


class Unet(nn.Module):
    def __init__(self, num_features=64, embedding_dim=128):
        super(Unet, self).__init__()
        self.num_features = num_features
        self.embedding_dim = embedding_dim
        self.time_embedding = SinusoidalPositionEmbeddings(embedding_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(embedding_dim, num_features),
            nn.GELU(),
            nn.Linear(num_features, num_features)
        )
        self.rrdb = RRDB(num_features=num_features)  # RRDB для обработки

    def forward(self, x, timesteps):
        # Кодирование временных шагов
        time_emb = self.time_embedding(timesteps)
        time_emb = self.time_mlp(time_emb).unsqueeze(-1).unsqueeze(-1)
        
        # Добавляем временной эмбеддинг к входным данным
        x = x + time_emb
        
        # Пропускаем через RRDB
        x = self.rrdb(x)
        
        return x
