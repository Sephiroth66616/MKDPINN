import torch
import torch.nn as nn

class HSM(nn.Module):
    def __init__(self, input_shape=(15, 14), d_model=15, num_heads= 3, dff=128,
                 dropout=0.1, output_dim=3):

        super(HSM, self).__init__()
        self.input_shape = input_shape

        # Embedding layer for feature dimension
        self.feature_embedding = nn.Linear(input_shape[1], d_model)

        # Transformer Encoder Layer
        encoder_layers = nn.TransformerEncoderLayer(d_model, num_heads, dff, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=2)

        # Output layer
        self.output = nn.Linear(d_model, output_dim)

    def forward(self, x):

        x = self.feature_embedding(x)
        x = x.permute(0, 2, 1)

        # Transformer Encoder
        x = self.transformer_encoder(x)

        # Mean pooling and output layer
        x = torch.mean(x, dim=1)
        x = self.output(x)
        return x

class PGR(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(PGR, self).__init__()
        self.hidden_dim = hidden_dim
        self.features = input_dim
        self.multihead_attn = nn.MultiheadAttention(self.features, 1)
        self.Dense1 = nn.Linear(self.features, self.features)
        self.Dense2 = nn.Linear(self.features, self.hidden_dim)
        self.LN = nn.LayerNorm(self.features)
        self.activation = nn.ReLU()

    def forward(self, X):
        x, weight = self.multihead_attn(X, X, X)
        x = self.LN(x + X)
        x1 = self.Dense1(x)
        x1 = self.activation(x1 + x)
        x1 = self.Dense2(x1)
        return x1


class PREDICTOR(nn.Module):
    def __init__(self, input_dim):
        super(PREDICTOR, self).__init__()
        self.features = input_dim
        self.params = nn.Parameter(torch.randn(8) * 0.01 + 100) #AWC
        self.dnn = nn.Sequential(
            nn.Linear(self.features, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 10),
            nn.Tanh(),
            nn.Linear(10, 1)
        )

    def forward(self, X):
        x = self.dnn(X)
        x = x * self.params
        return x.sum(dim=1)

