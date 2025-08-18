
import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedForexLSTM(nn.Module):
    """Enhanced LSTM with better architecture for forex prediction"""
    
    def __init__(self, input_size=20, hidden_size=128, num_layers=3, dropout=0.3):
        super(ImprovedForexLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Bidirectional LSTM layers with residual connections
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size,
            num_layers=1,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        
        self.lstm3 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size // 2,
            num_layers=1,
            dropout=0,
            batch_first=True,
            bidirectional=True
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            hidden_size,  # After concatenation
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head with focal loss support
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout // 2),
            nn.Linear(32, 3)  # Buy/Hold/Sell
        )
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.LayerNorm(hidden_size)
        ])
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, features = x.shape
        
        # Input normalization (reshape for batch norm)
        x_reshaped = x.reshape(-1, features)
        x_norm = self.input_norm(x_reshaped)
        x = x_norm.reshape(batch_size, seq_len, features)
        
        # LSTM layers with residual connections
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.layer_norms[0](lstm1_out)
        lstm1_out = self.dropout(lstm1_out)
        
        lstm2_out, _ = self.lstm2(lstm1_out)
        lstm2_out = self.layer_norms[1](lstm2_out)
        lstm2_out = self.dropout(lstm2_out)
        
        # Residual connection
        lstm2_out = lstm2_out + lstm1_out
        
        lstm3_out, _ = self.lstm3(lstm2_out)
        lstm3_out = self.layer_norms[2](lstm3_out)
        lstm3_out = self.dropout(lstm3_out)
        
        # Self-attention
        attended, _ = self.attention(lstm3_out, lstm3_out, lstm3_out)
        
        # Global average pooling + max pooling
        avg_pool = torch.mean(attended, dim=1)
        max_pool, _ = torch.max(attended, dim=1)
        
        # Combine pooled features
        combined = avg_pool + max_pool
        
        # Feature extraction
        features = self.feature_extractor(combined)
        
        # Classification
        logits = self.classifier(features)
        
        return F.softmax(logits, dim=1)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
