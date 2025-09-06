import torch
import torch.nn as nn

class EmbeddingLayer(nn.Module):
    """
    处理稀疏和数值特征的 Embedding 层
    """
    def __init__(self, categorical_field_dims, embed_dim):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(field_dim, embed_dim) for field_dim in categorical_field_dims
        ])

    def forward(self, x):
        # x: (batch_size, num_categorical_fields)
        # 输出: (batch_size, num_fields, embed_dim)
        embed_list = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        return torch.stack(embed_list, dim=1)
