# feedforward.py - CORRIGÉ
import torch
import torch.nn as nn
import torch.nn.functional as F

class FeedForward(nn.Module):
    """
    Feed-Forward Network (FFN) avec SwiGLU
    
    Architecture:
    - SwiGLU activation (meilleur que GELU)
    - Gate mechanism pour expressivité
    """
    def __init__(self, embed_dim, dropout=0.1, use_swiglu=True):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.use_swiglu = use_swiglu
        
        if use_swiglu:
            # SwiGLU avec 8/3 * embed_dim (compensation gate)
            self.hidden_dim = int(8 * embed_dim / 3)
            self.hidden_dim = (self.hidden_dim + 63) // 64 * 64
            
            self.gate_proj = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            self.up_proj = nn.Linear(embed_dim, self.hidden_dim, bias=False)
            self.down_proj = nn.Linear(self.hidden_dim, embed_dim, bias=False)
        else:
            # Fallback GELU
            self.hidden_dim = 4 * embed_dim
            self.fc1 = nn.Linear(embed_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        if self.use_swiglu:
            gate = F.silu(self.gate_proj(x))
            value = self.up_proj(x)
            x = gate * value
            x = self.down_proj(x)
            x = self.dropout(x)
        else:
            x = self.fc1(x)
            x = F.gelu(x)
            x = self.fc2(x)
            x = self.dropout(x)
        
        return x