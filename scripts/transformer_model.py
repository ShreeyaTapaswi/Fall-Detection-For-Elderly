import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class FallTransformer(nn.Module):
    def __init__(self, num_keypoints=17, d_model=128, nhead=4, num_layers=2, dim_feedforward=512, num_classes=5):
        super(FallTransformer, self).__init__()
        
        # Input: (batch, seq_len, 17*3) -> 17 keypoints * (x, y, conf)
        self.input_linear = nn.Linear(num_keypoints * 3, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc_out = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_linear(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        
        # We take the output of the last time step for classification
        # Alternatively, we could pool across time
        x = x[:, -1, :] # (batch_size, d_model)
        x = self.dropout(x)
        return self.fc_out(x)

if __name__ == "__main__":
    # Test the model
    model = FallTransformer()
    dummy_input = torch.randn(16, 30, 17*3) # Batch size 16, Seq len 30, 51 features
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expect (16, 3)
