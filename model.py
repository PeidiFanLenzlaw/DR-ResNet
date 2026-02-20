import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicRoutingBlock(nn.Module):
    """
    Dynamic Routing Block with adaptive residual connections.
    This module introduces a dynamic routing mechanism that adjusts feature flow 
    based on input importance, enhancing feature selection capability.
    
    Args:
        dim (int): Number of input/output channels
    """
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass with dynamic routing.
        
        Args:
            x (Tensor): Input tensor of shape [B, C, L]
            
        Returns:
            Tensor: Output tensor of shape [B, C, L]
        """
        residual = x
        x = F.gelu(self.conv1(x))
        gate = self.attn(x).unsqueeze(-1)
        return residual * (1 - gate) + x * gate


class DRResNet(nn.Module):
    """
    Dynamic Routing Residual Network (DR-ResNet).
    
    This network introduces dynamic routing mechanisms into residual connections,
    allowing adaptive feature flow based on input importance. The dynamic routing
    mechanism enhances feature selection capability while residual connections
    alleviate gradient vanishing. Group processing reduces computational cost.
    
    Architecture:
        1. Initial expansion layer: Projects input to higher dimensional space
        2. Dynamic routing blocks: 4 sequential blocks with adaptive routing
        3. Classifier: Maps features to class logits
    
    Args:
        input_dim (int): Dimension of input features (default: 3)
        hidden_dim (int): Hidden dimension for routing blocks (default: 16)
        num_classes (int): Number of output classes (default: 4)
    """
    def __init__(self, input_dim=3, hidden_dim=16, num_classes=4):
        super().__init__()
        self.init_expand = nn.Linear(input_dim, hidden_dim * 4)
        self.blocks = nn.Sequential(
            *[DynamicRoutingBlock(hidden_dim) for _ in range(4)]
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        Forward pass through DR-ResNet.
        
        Args:
            x (Tensor): Input tensor of shape [B, input_dim]
            
        Returns:
            Tensor: Class logits of shape [B, num_classes]
        """
        x = self.init_expand(x).unsqueeze(-1)  # [B, hidden_dim*4, 1]
        x = x.chunk(4, dim=1)[0]  # Group processing
        x = self.blocks(x).mean(dim=-1)  # [B, hidden_dim]
        return self.classifier(x)
