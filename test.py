import torch
import numpy as np
from yaml_config_manager import load_config
from tqdm import tqdm
import os

from model_classes.audio_classification_model import AudioClassificationModel
from additional_classes.checkpoint_manager import CheckpointManager

from utils import get_dataset, get_device, create_model, get_single_dataloader
from utils import get_classification_loss, compute_metrics, save_results_file, save_confusion_matrix


def evaluate_test_set(config, model, dataloader, device, criterions, return_embeddings=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_embeddings = []
    all_sample_types = []
    
    with torch.no_grad():
        p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing", leave=False)
        for i, batch in p_bar:
            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(batch)
            if config.model.num_classes == 2:
                # outputs = outputs.squeeze(1)
                logits = outputs["logits"].squeeze(1)
                targets = batch["labels"].float()
            else:
                logits = outputs["logits"]
                targets = batch["labels"]
            loss = criterions["classification"](logits, targets)
            running_loss += loss.item()
            
            if config.model.num_classes == 2:
                current_predictions = torch.sigmoid(logits).detach().cpu().numpy()
                current_predictions = np.where(current_predictions > 0.5, 1, 0)
            else:
                current_predictions = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
            
            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(current_predictions)
            
            if return_embeddings:
                all_embeddings.append(outputs["embeddings"].cpu().numpy())
                if "sample_type" in batch:
                    all_sample_types.extend(batch["sample_type"].cpu().numpy().tolist())
                else:
                    # Default to speech (0) if sample_type is not provided
                    all_sample_types.extend([0] * len(batch["labels"]))
                    print("Warning: 'sample_type' not found in batch. Defaulting to speech type (0).")
    
    metrics = compute_metrics(all_labels, all_predictions, is_binary_classification=config.model.num_classes == 2)
    metrics["loss"] = running_loss / len(dataloader)
    
    if return_embeddings:
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        return metrics, all_embeddings, all_labels, all_sample_types
    
    return metrics

def main(config):
    # Load the test datasets
    test_ewadb = get_dataset(config, "test", "ewadb", domain_id=0)
    test_pcgita = get_dataset(config, "test", "pc_gita", domain_id=1)
    
    print("Test datasets loaded successfully")
    print("Test EWADB dataset length: ", len(test_ewadb))
    print("Test PCGITA dataset length: ", len(test_pcgita))

    # Initialize device, model, and dataloader
    device = get_device(config)
    print(f"Using device: {device}")
    
    # set number of domains
    config.model.num_domains = 2
    
    model = create_model(config, device)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir=config.training.checkpoint_dir,
        model=model,
        optimizer=None,
        scheduler=None,
        device=device,
        lower_is_better=config.validation.metric_lower_is_better
    )
    
    # Load the best model from checkpoint
    checkpoint_manager.load_best_model()
    print("Loaded best model from checkpoint")
    
    # Create separate test dataloaders for each dataset
    test_dl_ewadb = get_single_dataloader(config, test_ewadb, "test")
    test_dl_pcgita = get_single_dataloader(config, test_pcgita, "test")
    
    criterions = {}
    criterions["classification"] = get_classification_loss(config.model.num_classes)
    # criterions["domain_classification"] = get_classification_loss(config.model.num_domains)
    
    
    # Evaluate the model on the test sets with embeddings
    test_metrics_ewadb, embeddings_ewadb, labels_ewadb, sample_types_ewadb = evaluate_test_set(config, model, test_dl_ewadb, device, criterions, return_embeddings=True)
    test_metrics_pcgita, embeddings_pcgita, labels_pcgita, sample_types_pcgita = evaluate_test_set(config, model, test_dl_pcgita, device, criterions, return_embeddings=True)
    
    print(f"[EWADB] Test Metrics:")
    for m in test_metrics_ewadb:
        print(f"Test {m}: {test_metrics_ewadb[m]}")
    
    print(f"[PCGITA] Test Metrics:")
    for m in test_metrics_pcgita:
        print(f"Test {m}: {test_metrics_pcgita[m]}")
    
    # Save the results and confusion matrices for both test datasets
    save_results_file(config.training.checkpoint_dir, test_metrics_ewadb, prefix="ewadb_")
    save_results_file(config.training.checkpoint_dir, test_metrics_pcgita, prefix="pcgita_")
    
    save_confusion_matrix(config.training.checkpoint_dir, test_metrics_ewadb["confusion_matrix"], prefix="ewadb_")
    save_confusion_matrix(config.training.checkpoint_dir, test_metrics_pcgita["confusion_matrix"], prefix="pcgita_")


if __name__ == "__main__":
    config = load_config()
    main(config)
