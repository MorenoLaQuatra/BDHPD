import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFeatureExtractor(nn.Module):
    def __init__(
        self, 
        conv_dims = [512, 512, 512, 512, 512, 512, 512],
        conv_kernels = [10, 3, 3, 3, 3, 2, 2],
        conv_strides = [5, 2, 2, 2, 2, 2, 2],
        final_dimension = 768,
        conv_bias=True,
        activation="gelu",
        dropout=0.1,
        layer_norm_eps=1e-5
    ):
        """
        Initialize the AudioFeatureExtractor.
        
        Args:
            conv_dims (list): List of output dimensions for each convolutional layer.
            conv_kernels (list): List of kernel sizes for each convolutional layer.
            conv_strides (list): List of strides for each convolutional layer.
            final_dimension (int): The final dimension of the extracted features.
            conv_bias (bool): Whether to use bias in the convolutional layers. Default is True.
            activation (str): Activation function to use. Default is "gelu".
            dropout (float): Dropout rate for the feature projection layer. Default is 0.1.
            layer_norm_eps (float): Epsilon for layer normalization. Default is 1e-5.
        """
        super(CNNFeatureExtractor, self).__init__()

        # Define activation function
        self.activation = self.get_activation_function(activation)

        # Define convolutional layers
        self.conv_layers = nn.ModuleList()
        in_conv_dim = 1  # input is a single-channel audio waveform
        for i in range(len(conv_dims)):
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_conv_dim,
                    out_channels=conv_dims[i],
                    kernel_size=conv_kernels[i],
                    stride=conv_strides[i],
                    bias=conv_bias
                )
            )
            in_conv_dim = conv_dims[i]

        # Feature projection layer to map to final_dimension
        self.layer_norm = nn.LayerNorm(conv_dims[-1], eps=layer_norm_eps)
        self.projection = nn.Linear(conv_dims[-1], final_dimension)
        self.dropout = nn.Dropout(dropout)

    def get_activation_function(self, activation):
        """
        Return the corresponding activation function.

        Args:
            activation (str): The name of the activation function.
        
        Returns:
            A callable activation function.
        """
        activation_functions = {
            "relu": F.relu,
            "gelu": F.gelu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
        }
        return activation_functions.get(activation, F.gelu)

    def forward(self, x):
        """
        Forward pass of the feature extractor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).
        
        Returns:
            torch.Tensor: Extracted features of shape (batch_size, num_features, final_dimension).
        """
        x = x.unsqueeze(1)  # Add a channel dimension for Conv1d (batch_size, 1, sequence_length)

        # Apply convolutional layers with activation
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
            x = self.activation(x)
        
        # B, C, T -> B, T, C
        x = x.transpose(1, 2)
        
        # Apply layer normalization and projection to final_dimension
        x = self.layer_norm(x)
        x = self.projection(x)
        x = self.dropout(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Parameters for 7 conv layers similar to Wav2Vec2
    conv_dims = [512, 512, 512, 512, 512, 512, 512]
    conv_kernels = [10, 3, 3, 3, 3, 2, 2]
    conv_strides = [5, 2, 2, 2, 2, 2, 2]
    final_dimension = 768

    # Initialize the feature extractor
    feature_extractor = CNNFeatureExtractor(conv_dims, conv_kernels, conv_strides, final_dimension, activation="gelu")

    # Dummy input: batch of audio waveforms
    input_waveform = torch.randn(4, 16000)  # (batch_size, sequence_length)

    # Extract features
    features = feature_extractor(input_waveform)
    print(features.shape)  # Output shape: (batch_size, feature_length, final_dimension)
