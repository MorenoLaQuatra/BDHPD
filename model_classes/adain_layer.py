import torch
import torch.nn as nn

class AdaIN(nn.Module):
    def __init__(self, embed_dim, use_embeddings=False, embedding_dim=None, num_domains=None, use_batch_stats=False):
        """
        Initialize the Adaptive Instance Normalization (AdaIN) layer.
        Args:
            embed_dim: The dimension of the input features (F).
            use_embeddings: Whether to condition AdaIN on embeddings (e.g., domain embeddings).
            embedding_dim: The dimension of the conditioning embeddings (if used).
            num_domains: The number of possible domain IDs (if using domain IDs).
            use_batch_stats: Whether to use batch statistics for normalization.
        """
        super(AdaIN, self).__init__()

        self.use_embeddings = use_embeddings
        self.num_domains = num_domains
        self.use_batch_stats = use_batch_stats

        if use_embeddings:
            assert embedding_dim is not None, "embedding_dim must be provided if use_embeddings is True"
            if num_domains is not None:
                self.domain_embeddings = nn.Embedding(num_embeddings=num_domains, embedding_dim=embedding_dim)
            self.fc_scale = nn.Linear(embedding_dim, embed_dim)
            self.fc_shift = nn.Linear(embedding_dim, embed_dim)
        else:
            self.scale = nn.Parameter(torch.ones(1, 1, embed_dim))
            self.shift = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if use_batch_stats:
            self.bn = nn.BatchNorm1d(embed_dim, affine=False)

    def forward(self, x, domain_id=None):
        """
        Forward pass of the AdaIN layer.
        Args:
            x: Input tensor of shape (batch_size, time_frames, feature_dim).
            domain_id: Optional domain IDs tensor of shape (batch_size,).
                       Required if use_embeddings is True and num_domains is provided.
        Returns:
            Output tensor of shape (batch_size, time_frames, feature_dim) after AdaIN.
        """
        if self.use_batch_stats:
            # Normalize using batch statistics
            x_reshaped = x.view(-1, x.shape[-1])
            x_norm = self.bn(x_reshaped).view(x.shape)
        else:
            # Calculate mean and standard deviation across the feature dimension
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True)
            x_norm = (x - mean) / (std + 1e-5)

        if self.use_embeddings:
            assert domain_id is not None, "domain_id must be provided if use_embeddings is True and num_domains is set"
            embeddings = self.domain_embeddings(domain_id)  # Shape: (B, embedding_dim)
            scale = self.fc_scale(embeddings).unsqueeze(1)  # Shape: (B, 1, embed_dim)
            shift = self.fc_shift(embeddings).unsqueeze(1)  # Shape: (B, 1, embed_dim)
        else:
            scale = self.scale  # Shape: (1, 1, embed_dim)
            shift = self.shift  # Shape: (1, 1, embed_dim)

        # Apply adaptive scaling and shifting
        x_out = scale * x_norm + shift

        return x_out

if __name__ == "__main__":
    # Example configuration
    batch_size = 64
    time_frames = 500
    feature_dim = 768
    embedding_dim = 128
    num_domains = 2

    # Create dummy input tensor with shape (batch_size, time_frames, feature_dim)
    input_tensor = torch.randn(batch_size, time_frames, feature_dim)
    print("Input shape:", input_tensor.shape)

    # Case 1: AdaIN without external embeddings or domain IDs, using batch statistics
    adain_layer = AdaIN(embed_dim=feature_dim, use_embeddings=False, use_batch_stats=True)
    output_tensor = adain_layer(input_tensor)
    print("Output shape (without embeddings, with batch stats):", output_tensor.shape)

    # Case 2: AdaIN with learned domain embeddings using domain IDs
    domain_ids = torch.randint(0, num_domains, (batch_size,))
    print("Domain IDs:", domain_ids)
    adain_layer_with_ids = AdaIN(embed_dim=feature_dim, use_embeddings=True, embedding_dim=embedding_dim, num_domains=num_domains)
    output_tensor_with_ids = adain_layer_with_ids(input_tensor, domain_ids)
    print("Output shape (with domain embeddings):", output_tensor_with_ids.shape)
