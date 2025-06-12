import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """ Self-attention Layer"""
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # 1x1 convolutions to project the input into Query, Key, and Value
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)
        
        self.softmax = nn.Softmax(dim=-1) # Softmax applied to the last dimension
        
        # A learnable scaling factor (gamma) for the attention output
        # Initialized to 0 so the network initially relies on local convolutions
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Project input to Q, K, V
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1) # B x (W*H) x C'
        proj_key = self.key_conv(x).view(batch_size, -1, width * height) # B x C' x (W*H)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height) # B x C x (W*H)

        # Calculate attention map (energy)
        energy = torch.bmm(proj_query, proj_key) # B x (W*H) x (W*H)
        attention = self.softmax(energy) # B x (W*H) x (W*H)
        
        # Apply attention to the value projection
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, width, height)
        
        # Apply the learned gamma and add the original input (residual connection)
        out = self.gamma * out + x
        return out