import torch
import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    """
    专家 (Expert) 层的 MLP
    """
    def __init__(self, input_dim, hidden_units, dropout=0.0, output_layer=True):
        super().__init__()
        layers = []
        for hidden_dim in hidden_units:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
        if output_layer:
            layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)
