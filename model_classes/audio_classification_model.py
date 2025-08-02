import torch
import torch.nn as nn
from transformers import AutoModel
from model_classes.conv_bottleneck_layer import ConvBottleneckLayer
from model_classes.cnn_feature_extractor import CNNFeatureExtractor
from model_classes.adain_layer import AdaIN


class AttentionPoolingLayer(nn.Module):
    """Implements attention pooling over a sequence of vectors.

    Args:
        embed_dim: The dimensionality of the input embedding.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the attention pooling layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim).

        Returns:
            Output tensor of shape (batch_size, embed_dim).
        """
        weights = torch.softmax(self.linear(x), dim=1)
        pooled_output = torch.sum(weights * x, dim=1)
        return pooled_output


class AudioClassificationModel(nn.Module):
    """Audio Classification Model incorporating SSL model and optional domain adaptation.

    Args:
        config: Configuration dictionary containing model parameters.
    """
    def __init__(self, config: dict):
        super().__init__()
        if config is None:
            raise ValueError("Configuration dictionary must be provided.")
        self.config = config

        self._initialize_model_params()
        self._initialize_ssl_model()
        self._initialize_wavelets_layer()
        self._initialize_cnn_feature_extractor()
        self._initialize_combined_layernorm()
        self._initialize_adain_layers()
        self._initialize_conv_bottleneck_layer()
        self._initialize_classification_heads()
        self._initialize_custom_parameters()

    def _initialize_model_params(self):
        """Initializes basic model parameters."""
        model_config = self.config.model
        data_config = self.config.data

        self.model_name_or_path = model_config.model_name_or_path
        self.num_classes = model_config.num_classes
        self.classifier_type = model_config.classifier_type
        self.num_layers = model_config.classifier_num_layers
        self.hidden_size = model_config.classifier_hidden_size
        self.dropout = model_config.dropout
        self.ssl = data_config.ssl
        self.feature_extractor = data_config.feature_extractor
        self.use_wavelets = data_config.wavelets
        self.use_adain_layers = model_config.use_adain_layers
        self.use_conv_bottleneck = model_config.use_conv_bottleneck_layer
        
        print(f"[INFO] Using SSL model: {self.ssl}")
        print(f"[INFO] Using wavelets: {self.use_wavelets}")

        self.pooling_embedding_dim = 0
        self.global_embedding_dim = 0

    def _initialize_ssl_model(self):
        """Initializes the SSL model and related components."""
        if not self.ssl:
            return

        self.ssl_model = self._load_ssl_model()
        self.pooling_embedding_dim += self.ssl_model.config.hidden_size
        self.global_embedding_dim += self.ssl_model.config.hidden_size
        self.after_ssl_layer_norm = nn.LayerNorm(self.pooling_embedding_dim)

    def _initialize_wavelets_layer(self):
        """Initializes the wavelets layer if configured."""
        if self.use_wavelets:
            spec_dim = self.config.data.stft_params.spec_dim
            self.pooling_embedding_dim += spec_dim
            self.global_embedding_dim += spec_dim
            self.after_wavelets_layer_norm = nn.LayerNorm(spec_dim)
            
    def _initialize_cnn_feature_extractor(self):
        """Initializes the CNN feature extractor."""
        if not self.ssl:
            print("[INFO] SSL model not used. Initializing CNN feature extractor.")
            self.cnn_feature_extractor = CNNFeatureExtractor(
                final_dimension=self.config.model.classifier_hidden_size,
            )
            self.pooling_embedding_dim += self.config.model.classifier_hidden_size
            self.global_embedding_dim += self.config.model.classifier_hidden_size
        else:
            self.cnn_feature_extractor = None
            
    def _initialize_combined_layernorm(self):
        """Initializes the combined layer normalization layer."""
        self.after_combined_layer_norm = None
            
    def _initialize_adain_layers(self):
        """Initializes the AdaIN layers for both SSL and wavelet features if configured."""
        if self.use_adain_layers:
            if self.ssl:
                self.adain_ssl = nn.ModuleList([
                    AdaIN(embed_dim=self.ssl_model.config.hidden_size,
                        use_embeddings=True,
                        embedding_dim=self.config.model.adain_embedding_dim,
                        num_domains=self.config.model.num_domains)
                    for _ in range(self.config.model.adain_num_layers)
                ])
            else:
                self.adain_ssl = None

            if self.use_wavelets:
                self.adain_wavelets = nn.ModuleList([
                    AdaIN(embed_dim=self.config.data.stft_params.spec_dim,
                        use_embeddings=True,
                        embedding_dim=self.config.model.adain_embedding_dim,
                        num_domains=self.config.model.num_domains)
                    for _ in range(self.config.model.adain_num_layers)
                ])
            else:
                self.adain_wavelets = None
        else:
            self.adain_ssl = None
            self.adain_wavelets = None
            
    def _initialize_conv_bottleneck_layer(self):
        """Initializes the Convolutional Bottleneck layer if configured."""
        if self.use_conv_bottleneck:
            self.conv_bottleneck_layers = nn.ModuleList([
                ConvBottleneckLayer(
                    embed_dim=self.pooling_embedding_dim,
                    reduction=self.config.model.conv_bottleneck_reduction_ratio
                ) for _ in range(self.config.model.conv_bottleneck_num_layers)
            ])
        else:
            self.conv_bottleneck_layers = None

    def _load_ssl_model(self) -> nn.Module:
        """Loads the SSL model based on the configuration."""
        model = AutoModel.from_pretrained(
            self.model_name_or_path, 
            output_hidden_states=self.config.model.use_all_layers
        )
        if "whisper" in self.model_name_or_path:
            model = model.encoder
            self.is_whisper = True
        else:
            self.is_whisper = False

        if self.config.model.increase_resolution_cnn:
            model.feature_extractor.conv_layers[6].conv.stride = (1,)

        if self.config.model.freeze_ssl:
            for param in model.parameters():
                param.requires_grad = False

        if self.config.model.use_all_layers:
            num_layers = model.config.num_hidden_layers + 1
            self.layer_weights = nn.Parameter(torch.ones(num_layers))
            self.layer_weights.requires_grad = True
            self.layer_norms = nn.ModuleList([nn.LayerNorm(model.config.hidden_size) for _ in range(num_layers)])
            self.layer_norms.requires_grad = True
            self.softmax = nn.Softmax(dim=-1)

        return model

    def _initialize_classification_heads(self):
        """Initializes the classification and domain adaptation heads."""
        if self.config.model.type_based_classifiers:
            print("\n\n[INFO] Using type-based classifiers.\n\n")
            self.branch_layers = nn.ModuleList()
            self.pooling_layers = nn.ModuleList()
            self.classifiers = nn.ModuleList()
            
            for _ in range(self.config.model.num_class_types):
                branch, clf, pool = self._create_classification_head(
                    self.pooling_embedding_dim, self.global_embedding_dim, self.num_classes
                )
                self.branch_layers.append(branch)
                self.pooling_layers.append(pool)
                self.classifiers.append(clf)
        else:
            self.branch, self.classifier, self.pooling_layer = self._create_classification_head(
                self.pooling_embedding_dim, self.global_embedding_dim, self.num_classes
            )

    def _create_pooling_layer(self, input_dim: int) -> nn.Module:
        """Creates the pooling layer based on the configuration."""
        if self.classifier_type == "attention_pooling":
            return AttentionPoolingLayer(input_dim)
        return None
        
    def _create_classification_head(self, pooling_input_dim: int, global_input_dim: int, num_classes: int, need_pooling: bool = True):
        """Creates a classification head with optional pooling.
        
        Args:
            pooling_input_dim: Input dimension for pooling.
            global_input_dim: Input dimension for the global layer.
            num_classes: Number of output classes.
            need_pooling: Whether to include pooling layer.
        
        Returns:
            Tuple of (classifier, pooling_layer).
        """
        branch_layers = nn.Sequential()
        pooling_layer = nn.Sequential()
        classifier = nn.Sequential()
        
        if self.classifier_type == "attention_pooling" and need_pooling:
            pooling_layer.add_module("attention_pooling", AttentionPoolingLayer(self.hidden_size))

        self._add_classifier_layers(branch_layers, classifier, global_input_dim, num_classes)
        
        if need_pooling:
            return branch_layers, classifier, pooling_layer
        return branch_layers, classifier
        
    def _add_classifier_layers(self, branch_layers: nn.Sequential, classifier: nn.Sequential, input_dim: int, num_classes: int):
        """Adds the necessary layers to the branch_layers and classifier."""
        if self.config.model.classifier_head_type == "linear":
            self._add_linear_layers(branch_layers, input_dim)
        elif self.config.model.classifier_head_type == "transformer":
            self._add_transformer_layers(branch_layers, input_dim)

        self._add_final_classifier_layer(classifier, num_classes)

    def _add_linear_layers(self, branch_layers: nn.Sequential, input_dim: int):
        """Adds linear layers to the branch_layers."""
        for layer in range(self.num_layers):
            input_size = input_dim if layer == 0 else self.hidden_size
            branch_layers.add_module(f"layer_{layer}", nn.Linear(input_size, self.hidden_size))
            branch_layers.add_module(f"activation_{layer}", nn.ReLU())
            branch_layers.add_module(f"dropout_{layer}", nn.Dropout(self.dropout))

    def _add_transformer_layers(self, branch_layers: nn.Sequential, input_dim: int):
        """Adds transformer layers to the branch_layers."""
        branch_layers.add_module("input_layer", nn.Linear(input_dim, self.hidden_size))
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.config.model.transformer_nhead,
            dim_feedforward=self.config.model.transformer_dim_feedforward,
            dropout=self.dropout,
            activation="relu",
            batch_first=True,
        )
        branch_layers.add_module(
            "transformer", 
            nn.TransformerEncoder(
                encoder_layer=transformer_layer,
                num_layers=self.num_layers
            )
        )

    def _add_final_classifier_layer(self, classifier: nn.Sequential, num_classes: int):
        """Adds the final output layer to the classifier."""
        final_layer = nn.Linear(self.hidden_size, 1 if num_classes == 2 else num_classes)
        classifier.add_module("final_layer", final_layer)

        if num_classes == 2 and self.config.model.apply_final_sigmoid:
            classifier.add_module("sigmoid_activation", nn.Sigmoid())

    def _initialize_custom_parameters(self):
        """Initializes custom parameters for layers outside the SSL model."""
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
        for name, module in self.named_modules():
            if 'ssl_model' not in name:
                init_weights(module)

    def _get_ssl_features(self, input_values: torch.Tensor) -> torch.Tensor:
        """Extracts features from the SSL model.
        
        Args:
            input_values: Input tensor for the SSL model.
        
        Returns:
            Extracted features tensor.
        """
        if self.config.model.use_all_layers:
            return self._get_combined_ssl_features(input_values)
        return self.ssl_model(input_values=input_values, return_dict=True).last_hidden_state

    def _get_combined_ssl_features(self, input_values: torch.Tensor) -> torch.Tensor:
        """Combines features from all layers of the SSL model."""
        hidden_states = self.ssl_model(input_values=input_values, return_dict=True).hidden_states
        combined_hidden_state = torch.zeros_like(hidden_states[-1])
        weights = self.softmax(self.layer_weights)

        for i, hidden_state in enumerate(hidden_states):
            combined_hidden_state += weights[i] * self.layer_norms[i](hidden_state)

        return combined_hidden_state

    def _combine_features(self, ssl_features: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """Combines SSL features with magnitudes."""
        return torch.cat([ssl_features, magnitudes], dim=-1)

    def forward(self, batch: dict) -> dict:
        """Forward pass of the model.
        
        Args:
            batch: Dictionary containing input features and other batch data.
        
        Returns:
            Dictionary containing logits and embeddings.
        """
        features = None
        ssl_features = None
        wavelet_features = None
        
        if self.ssl:
            ssl_input = batch["input_features"] if self.is_whisper else batch["input_values"]
            ssl_features = self._get_ssl_features(ssl_input)
            ssl_features = self.after_ssl_layer_norm(ssl_features)
        else:
            ssl_features = self.cnn_feature_extractor(batch["input_values"])

        if self.use_wavelets:
            wavelet_features = self.after_wavelets_layer_norm(batch["magnitudes"])
                
        if self.adain_ssl is not None:
            for adain_layer in self.adain_ssl:
                ssl_features = adain_layer(ssl_features, batch["domain_labels"])
                
        if self.adain_wavelets is not None:
            for adain_layer in self.adain_wavelets:
                wavelet_features = adain_layer(wavelet_features, batch["domain_labels"])
        
        if ssl_features is not None and wavelet_features is not None:
            features = self._combine_features(ssl_features, wavelet_features)
        elif ssl_features is not None:
            features = ssl_features
        elif wavelet_features is not None:
            features = wavelet_features
                
        if self.conv_bottleneck_layers is not None:
            for bottleneck_layer in self.conv_bottleneck_layers:
                features = bottleneck_layer(features)

        if self.config.model.type_based_classifiers:
            output_list = [None] * len(batch["sample_type"])
            embedding_list = [None] * len(batch["sample_type"])
            
            unique_sample_types = torch.unique(batch["sample_type"])
            
            for sample_type in unique_sample_types:
                type_indices = torch.where(batch["sample_type"] == sample_type)[0]
                type_features = features[type_indices]
                
                branch_layer = self.branch_layers[batch["sample_type"][type_indices[0]]]
                pooling_layer = self.pooling_layers[batch["sample_type"][type_indices[0]]]
                classifier = self.classifiers[batch["sample_type"][type_indices[0]]]
                
                type_features = branch_layer(type_features)
                type_features = torch.mean(type_features, dim=1) if self.classifier_type == "average_pooling" else pooling_layer(type_features)
                type_output = classifier(type_features)
                
                for idx, output in zip(type_indices, type_output):
                    output_list[idx.item()] = output
                for idx, embedding in zip(type_indices, type_features):
                    embedding_list[idx.item()] = embedding
            
            output = torch.stack(output_list)
            embeddings = torch.stack(embedding_list)
        else:
            features = self.branch(features)
            features = torch.mean(features, dim=1) if self.classifier_type == "average_pooling" else self.pooling_layer(features)
            output = self.classifier(features)
            embeddings = features
            
        results_dict = {"logits": output, "embeddings": embeddings}
            
        return results_dict