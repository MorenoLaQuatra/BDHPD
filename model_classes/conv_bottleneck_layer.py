import torch
import torch.nn as nn

class ConvBottleneckLayer(nn.Module):
    def __init__(self, embed_dim, reduction=16):
        """
        Initialize the Convolutional Bottleneck Layer.
        Args:
            embed_dim: The dimension of the input features (embed_dim).
            reduction: The reduction ratio for the bottleneck.
        """
        super(ConvBottleneckLayer, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=embed_dim // reduction, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=embed_dim // reduction, out_channels=embed_dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        """
        Forward pass of the Convolutional Bottleneck Layer.
        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).
        Returns:
            Output tensor of shape (batch_size, seq_len, embed_dim).
        """
        x_permuted = x.permute(0, 2, 1)
        
        x_out = self.conv1(x_permuted)
        x_out = self.relu(x_out)
        x_out = self.conv2(x_out)
        x_out = self.sigmoid(x_out)
        
        x_recalibrated = x_permuted * x_out + x_permuted
        x_recalibrated = x_recalibrated.permute(0, 2, 1)
        
        return x_recalibrated

if __name__ == "__main__":
    # Simple usage example
    batch_size, seq_len, embed_dim = 64, 500, 768
    input_tensor = torch.randn(batch_size, seq_len, embed_dim)
    bottleneck_layer = ConvBottleneckLayer(embed_dim=embed_dim)
    output_tensor = bottleneck_layer(input_tensor)
    print(f"Input shape: {input_tensor.shape}, Output shape: {output_tensor.shape}")
