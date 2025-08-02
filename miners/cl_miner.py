import torch
from pytorch_metric_learning.miners import BaseMiner
from pytorch_metric_learning.distances import LpDistance

class CLMiner(BaseMiner):
    def __init__(self, sample_within_domain_only=False, distance=None, **kwargs):
        """A custom miner for extracting hard positive and negative pairs based on sample types and domains.

        Args:
            sample_within_domain_only (bool): If True, positive and negative pairs are mined within the same domain.
            distance (Distance): The distance metric used for computing similarity. Defaults to LpDistance.
            **kwargs: Additional arguments for the BaseMiner.
        """
        super().__init__(**kwargs)
        self.sample_within_domain_only = sample_within_domain_only
        self.distance = distance if distance is not None else LpDistance(normalize_embeddings=True, p=2, power=1)

    def mine(self, embeddings, labels, sample_types, domains):
        """Mines hard positive and negative pairs based on sample types and domains.

        Args:
            embeddings (Tensor): The embedding vectors for the current batch (batch_size, embedding_dim).
            labels (Tensor): The ground truth labels for the current batch (batch_size,).
            sample_types (Tensor): Tensor indicating the type of each sample in the batch (batch_size,).
            domains (Tensor): Tensor indicating the domain of each sample in the batch (batch_size,).

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: Tuple containing indices for anchors, positives, 
                                                  and negatives for contrastive learning.
        """
        unique_types = torch.unique(sample_types)
        all_anchors, all_positives, all_negatives = [], [], []

        for sample_type in unique_types:
            type_indices = torch.where(sample_types == sample_type)[0]
            
            if len(type_indices) < 2:
                continue

            domain_indices = self._get_domain_indices(type_indices, domains) if self.sample_within_domain_only else type_indices

            type_output = self._mine_hard_pairs(
                embeddings[type_indices], 
                labels[type_indices], 
                embeddings[domain_indices], 
                labels[domain_indices]
            )

            if len(type_output[0]) == 0 or len(type_output[1]) == 0 or len(type_output[2]) == 0:
                continue

            anchors = type_indices[type_output[0]]
            positives = domain_indices[type_output[1]]
            negatives = domain_indices[type_output[2]]

            all_anchors.append(anchors)
            all_positives.append(positives)
            all_negatives.append(negatives)

        if not all_anchors or not all_positives or not all_negatives:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)

        all_anchors = torch.cat(all_anchors)
        all_positives = torch.cat(all_positives)
        all_negatives = torch.cat(all_negatives)

        min_len = min(len(all_anchors), len(all_positives), len(all_negatives))
        all_anchors = all_anchors[:min_len]
        all_positives = all_positives[:min_len]
        all_negatives = all_negatives[:min_len]

        return all_anchors, all_positives, all_anchors, all_negatives

    def _mine_hard_pairs(self, type_embeddings, type_labels, domain_embeddings, domain_labels):
        """Mines the hardest positive and negative pairs based on distances between embeddings.

        Args:
            type_embeddings (Tensor): Embeddings for a specific sample type.
            type_labels (Tensor): Labels for a specific sample type.
            domain_embeddings (Tensor): Embeddings for the domain under consideration.
            domain_labels (Tensor): Labels for the domain under consideration.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Indices of anchors, hardest positives, and hardest negatives.
        """
        distances = self.distance(type_embeddings, domain_embeddings)

        pos_pairs, neg_pairs = [], []
        for i in range(len(type_labels)):
            pos_mask = (type_labels[i] == domain_labels).float()
            neg_mask = 1 - pos_mask

            if pos_mask.sum() > 0:
                pos_distances = distances[i] * pos_mask
                hardest_positive = torch.argmax(pos_distances)
                pos_pairs.append((i, hardest_positive))

            if neg_mask.sum() > 0:
                neg_distances = distances[i] + (1e9 * pos_mask)
                hardest_negative = torch.argmin(neg_distances)
                neg_pairs.append((i, hardest_negative))

        if pos_pairs:
            pos_pairs = torch.tensor(pos_pairs)
            anchors = pos_pairs[:, 0]
            positives = pos_pairs[:, 1]
        else:
            anchors = torch.tensor([], dtype=torch.long)
            positives = torch.tensor([], dtype=torch.long)

        if neg_pairs:
            neg_pairs = torch.tensor(neg_pairs)
            negatives = neg_pairs[:, 1]
        else:
            negatives = torch.tensor([], dtype=torch.long)

        return anchors, positives, negatives

    def _get_domain_indices(self, type_indices, domains):
        """Gets indices of samples within the same domain as the specified type indices.

        Args:
            type_indices (Tensor): Indices of samples for a specific type.
            domains (Tensor): Domain labels for all samples.

        Returns:
            Tensor: Indices of samples within the same domain(s) as the specified type indices.
        """
        type_domains = domains[type_indices]
        unique_domains = torch.unique(type_domains)
        domain_indices = torch.cat([torch.where(domains == d)[0] for d in unique_domains])
        return domain_indices


if __name__ == "__main__":
    from pytorch_metric_learning import losses

    # Example setup for testing
    batch_size = 64
    embedding_dim = 768

    # Generate dummy data
    embeddings = torch.randn(batch_size, embedding_dim)
    labels = torch.randint(0, 10, (batch_size,))  # 10 different classes
    sample_types = torch.randint(0, 3, (batch_size,))  # 3 different sample types
    domains = torch.randint(0, 2, (batch_size,))  # 2 different domains (e.g., 0 = Spanish, 1 = Slovak)

    # Initialize the miner
    miner = CLMiner(sample_within_domain_only=True)
    loss_func = losses.ContrastiveLoss()

    # Mine pairs and compute the loss
    miner_output = miner.mine(embeddings, labels, sample_types, domains)
    loss = loss_func(embeddings, labels, miner_output)
    
    print("Embeddings shape:", embeddings.shape)
    print("Labels shape:", labels.shape)
    print("Sample types shape:", sample_types.shape)
    print("Domains shape:", domains.shape)

    # Output for verification
    print("Miner Output (anchors, positives, negatives):")
    print(miner_output)
    print ("anchors shape:", miner_output[0].shape)
    print ("positives shape:", miner_output[1].shape)
    print ("negatives shape:", miner_output[2].shape)
    print("Loss:", loss.item())
