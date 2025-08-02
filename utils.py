import os
from comet_ml import Experiment as CometExperiment
from additional_classes.terminal_experiment import TerminalExperiment
import torch
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)

from torch.utils.data import DataLoader
from additional_classes.lr_scheduler import LRSchedulerWithWarmup
from model_classes.audio_classification_model import AudioClassificationModel
from data_classes.ewadb_dataset import EWADBDataset
import matplotlib.pyplot as plt
from torch.utils.data import WeightedRandomSampler
from collections import Counter

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from tqdm import tqdm

def compute_metrics(reference, predictions, verbose=False, is_binary_classification=False):
    # Calculate basic metrics
    accuracy = accuracy_score(reference, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(reference, predictions, average="macro")
    
    # Initialize binary classification specific metrics
    roc_auc = 0.0
    sensitivity = 0.0
    specificity = 0.0
    
    if is_binary_classification:
        # Compute ROC AUC
        try:
            roc_auc = roc_auc_score(reference, predictions)
        except ValueError:
            if verbose:
                print("ROC AUC gave ValueError, setting to 0.0")
        
        # Compute confusion matrix metrics
        cm = confusion_matrix(reference, predictions)
        if cm.shape == (2, 2):
            tp, fn = cm[1, 1], cm[1, 0]
            tn, fp = cm[0, 0], cm[0, 1]
            sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) != 0 else 0.0
        else:
            if verbose:
                print("Confusion matrix shape is not (2,2) - maybe the model predicted only one class")
    else:
        if verbose:
            print("ROC AUC is not defined for multiclass classification")
            
            
    # print confusion matrix
    print(f"\n\n -- Confusion Matrix -- \n {cm} \n\n")
        
    # Optionally print metrics
    if verbose:
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")
        print(f"ROC AUC: {roc_auc}")
        print(f"Sensitivity: {sensitivity}")
        print(f"Specificity: {specificity}")
        
    # Return metrics as a dictionary
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
    }

def get_dataset(config, ds_type, dataset_name=None, domain_id=None):
    
    if dataset_name is None:
        dataset_name = config.active_dataset
        
    
    if dataset_name == "ewadb":
        metadata_type = config.ewadb.metadata_type
        dataset_root_path = config.ewadb.dataset_root_path
        audio_path_key = config.ewadb.audio_path_key
        label_key = config.ewadb.label_key
        label2id = config.ewadb.label2id
        if ds_type == "train": metadata_path = config.ewadb.train_metadata_path
        elif ds_type == "validation": metadata_path = config.ewadb.validation_metadata_path
        elif ds_type == "test": metadata_path = config.ewadb.test_metadata_path
        else: raise ValueError(f"Dataset type {ds_type} not supported")
        config.model.num_classes = 2
    elif dataset_name == "pc_gita":
        metadata_type = config.pc_gita.metadata_type
        dataset_root_path = config.pc_gita.dataset_root_path
        audio_path_key = config.pc_gita.audio_path_key
        label_key = config.pc_gita.label_key
        label2id = config.pc_gita.label2id
        if ds_type == "train": metadata_path = config.pc_gita.train_metadata_path
        elif ds_type == "validation": metadata_path = config.pc_gita.validation_metadata_path
        elif ds_type == "test": metadata_path = config.pc_gita.test_metadata_path
        else: raise ValueError(f"Dataset type {ds_type} not supported")
        config.model.num_classes = 2
    else:
        raise ValueError(f"Requested {dataset_name}, but it seems not supported")
    
    dataset = EWADBDataset(
        config=config,
        metadata_path=metadata_path,
        metadata_type=metadata_type,
        dataset_root_path=dataset_root_path,
        audio_path_key=audio_path_key,
        label_key=label_key,
        model_name_or_path=config.model.model_name_or_path,
        label2id=label2id,
        domain_id=domain_id,
        is_test=False if ds_type == "train" else True,
    )
    return dataset

def create_optimizer_and_scheduler(model, config, train_dataset_length):
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    
    total_steps = int(train_dataset_length / config.training.batch_size * config.training.num_epochs)
    
    scheduler = LRSchedulerWithWarmup(
        optimizer,
        total_steps=total_steps,
        decay_type=config.training.scheduler_decay_type,
        warmup_ratio=config.training.warmup_ratio,
    )
    
    return optimizer, scheduler

def get_device(config):
    if torch.cuda.is_available() and config.training.use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    return device

def compute_sample_weights(dataset, label_key="labels"):
    print("Computing sample weights...")
    try:
        labels = dataset.get_list_labels()
    except AttributeError:
        print("Dataset does not have get_list_labels method, using slower method")
        labels = np.array([sample[label_key] for sample in tqdm(dataset)])
        if isinstance(labels[0], torch.Tensor):
            labels = labels.numpy()
            
    label_counts = np.bincount(labels)
    label_weights = 1.0 / label_counts
    sample_weights = label_weights[labels]
    # print(f"Sample weights computed: {sample_weights}")
    # print ALL weights 10 by 10, together with labels
    # for i in range(0, len(sample_weights), 10):
    #     print(f"Sample weights: {sample_weights[i:i+10]}")
    #     print(f"Labels: {labels[i:i+10]}")
    return sample_weights

def get_single_dataloader(config, dataset, ds_type, balance_dataloader=False):
    if balance_dataloader:
        sample_weights = compute_sample_weights(dataset)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    else:
        sampler = None

    dataloader = DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=not sampler and ds_type == "train",
        num_workers=config.training.num_workers,
        pin_memory=True,
        sampler=sampler,
        worker_init_fn=lambda _: set_all_seeds_for_reproducibility()
    )
    
    print(f"Created {ds_type} dataloader with {len(dataloader)} batches")
    
    return dataloader

def create_model(config, device):
    model = AudioClassificationModel(config)
    # num active params
    active_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of active parameters: {active_params / 1e6 :.2f}M / {total_params / 1e6:.2f}M ({active_params / total_params * 100:.2f}%)")
    model = model.to(device)
    return model

def resume_from_checkpoint_if_needed(config, checkpoint_manager):
    
    if config.training.resume_from_last_checkpoint:
        checkpoint_path = checkpoint_manager.get_latest_checkpoint()
        if checkpoint_path is not None:
            start_epoch, best_metric = checkpoint_manager.load_checkpoint(checkpoint_path, load_optimizer_scheduler=True)
            print(f"Resuming training from epoch {start_epoch} with best metric {best_metric}")
        else:
            print("No checkpoint found to resume training")
            start_epoch = 0
            best_metric = 1e9 if config.validation.metric_lower_is_better else -1e9
    else:
        start_epoch = 0
        best_metric = 1e9 if config.validation.metric_lower_is_better else -1e9
        
    return start_epoch, best_metric
    
            
def get_experiment(config):
    if config.comet.use_comet:
        experiment = CometExperiment(
            api_key=os.environ.get("COMET_API_KEY"),
            project_name=config.training.comet_project_name,
            workspace=os.environ.get("COMET_WORKSPACE"),
        )
        experiment.set_name(config.training.come_experiment_name)
        experiment.log_parameters(config)
    else:
        experiment = TerminalExperiment(log_on_screen=config.training.log_on_screen)
        
    return experiment

def get_classification_loss(num_classes):
    # check if binary classification
    if num_classes == 2:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    return criterion

def save_results_file(checkpoint_dir, test_metrics, prefix=""):
    with open(os.path.join(checkpoint_dir, f"{prefix}test_results.tsv"), "w") as f:
        f.write("metric\tvalue\n")
        for m in test_metrics:
            try:
                f.write(f"{m}\t{test_metrics[m]*100:.2f}\n")
            except Exception as e:
                print(f"Error writing metric {m} to file: {e}")
        f.write(f"{test_metrics['accuracy']*100:.2f}\t{test_metrics['precision']*100:.2f}\t{test_metrics['recall']*100:.2f}\t{test_metrics['f1']*100:.2f}\t{test_metrics['roc_auc']*100:.2f}\t{test_metrics['sensitivity']*100:.2f}\t{test_metrics['specificity']*100:.2f}\n")
            
    print(f"Results saved to {os.path.join(checkpoint_dir, f'{prefix}test_results.tsv')}")
            
def save_confusion_matrix(checkpoint_dir, cm, labels=None, prefix=""):
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, cmap="Blues")
    
    # Adding a color bar
    plt.colorbar()
    
    # Adding titles and labels with larger font sizes
    plt.xlabel("Predicted", fontsize=16)
    plt.ylabel("True", fontsize=16)
    plt.title("Confusion Matrix", fontsize=20)
    
    # Add the labels for x and y axis if provided
    if labels is not None:
        plt.xticks(np.arange(len(labels)), labels, fontsize=14)
        plt.yticks(np.arange(len(labels)), labels, fontsize=14)
    else:
        plt.xticks(np.arange(len(cm)), np.arange(len(cm)), fontsize=14)
        plt.yticks(np.arange(len(cm)), np.arange(len(cm)), fontsize=14)

    # Display the percentage values within each class (row)
    for i in range(cm.shape[0]):
        row_sum = np.sum(cm[i, :])
        for j in range(cm.shape[1]):
            if row_sum > 0:
                percentage = cm[i, j] / row_sum * 100
            else:
                percentage = 0
            plt.text(j, i, f'{cm[i, j]}\n({percentage:.2f}%)',
                     horizontalalignment='center',
                     color='white' if cm[i, j] > np.max(cm) / 2 else 'black',
                     fontsize=14)
    
    # Save the confusion matrix as a PNG file
    plt.savefig(os.path.join(checkpoint_dir, f"{prefix}confusion_matrix.png"), bbox_inches='tight')
    print(f"Confusion matrix saved to {os.path.join(checkpoint_dir, f'{prefix}confusion_matrix.png')}")
    
def plot_embeddings(checkpoint_dir, embeddings, labels, sample_types, method="TSNE", prefix="", class_mapping={0: "Healthy", 1: "Parkinson"}, sample_type_mapping=None):
    if sample_type_mapping is None:
        sample_type_mapping = {0: "Speech", 1: "DDK"}  # Default mapping if none provided
    
    # Ensure embeddings is a numpy array
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)
    
    # Reduce dimensions to 2D
    if method.upper() == "PCA":
        reducer = PCA(n_components=2)
    elif method.upper() == "TSNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be either 'PCA' or 'TSNE'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot settings
    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    unique_sample_types = np.unique(sample_types)
    
    # Colorblind-friendly colors
    colorblind_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']  
    shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']  # Different marker shapes for sample types

    for i, label in enumerate(unique_labels):
        for j, sample_type in enumerate(unique_sample_types):
            indices = np.where((labels == label) & (sample_types == sample_type))
            color = colorblind_colors[i % len(colorblind_colors)]  # Color based on label
            shape = shapes[j % len(shapes)]  # Shape based on sample type
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                        marker=shape, color=color, 
                        label=f'{class_mapping[label]} ({sample_type_mapping[sample_type]})', 
                        s=100, edgecolor='black', alpha=0.7)

    # Add legend with both colors and shapes
    legend_elements = [Line2D([0], [0], marker=shapes[j % len(shapes)], color='w', 
                              label=f'{class_mapping[label]} ({sample_type_mapping[sample_type]})',
                              markerfacecolor=colorblind_colors[i % len(colorblind_colors)], markersize=10, markeredgecolor='black') 
                       for i, label in enumerate(unique_labels)
                       for j, sample_type in enumerate(unique_sample_types)]
    
    plt.legend(handles=legend_elements, fontsize=12, loc='best')
    
    plt.title(f'{method} of Embeddings', fontsize=20)
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.grid(True)  # Add grid

    # Save the plot
    output_path = os.path.join(checkpoint_dir, f"{prefix}embeddings.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Embeddings plot saved to {output_path}")


from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting

def plot_embeddings_3d(checkpoint_dir, embeddings, labels, sample_types, method="PCA", prefix="", class_mapping={0: "Healthy", 1: "Parkinson"}, sample_type_mapping=None):
    if sample_type_mapping is None:
        sample_type_mapping = {0: "Speech", 1: "DDK"}  # Default mapping if none provided
    
    # Ensure embeddings is a numpy array
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    # Debugging: Check the shape of embeddings
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Reduce dimensions to 2D
    if method.upper() == "PCA":
        reducer = PCA(n_components=2)
    elif method.upper() == "TSNE":
        reducer = TSNE(n_components=2, random_state=42)
    else:
        raise ValueError("Method must be either 'PCA' or 'TSNE'.")

    reduced_embeddings = reducer.fit_transform(embeddings)

    # Plot settings
    plt.figure(figsize=(12, 10))

    unique_labels = np.unique(labels)
    unique_sample_types = np.unique(sample_types)
    
    # Colorblind-friendly colors
    colorblind_colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#F0E442']  
    shapes = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']  # Different marker shapes for sample types

    for i, label in enumerate(unique_labels):
        for j, sample_type in enumerate(unique_sample_types):
            indices = np.where((labels == label) & (sample_types == sample_type))
            color = colorblind_colors[i % len(colorblind_colors)]  # Color based on label
            shape = shapes[j % len(shapes)]  # Shape based on sample type
            plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], 
                        marker=shape, color=color, 
                        label=f'{class_mapping[label]} ({sample_type_mapping[sample_type]})', 
                        s=100, edgecolor='black', alpha=0.7)

    # Add legend with both colors and shapes
    legend_elements = [Line2D([0], [0], marker=shapes[j % len(shapes)], color='w', 
                              label=f'{class_mapping[label]} ({sample_type_mapping[sample_type]})',
                              markerfacecolor=colorblind_colors[i % len(colorblind_colors)], markersize=10, markeredgecolor='black') 
                       for i, label in enumerate(unique_labels)
                       for j, sample_type in enumerate(unique_sample_types)]
    
    plt.legend(handles=legend_elements, fontsize=12, loc='best')
    
    plt.title(f'{method} of Embeddings', fontsize=20)
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.grid(True)  # Add grid

    # Save the plot
    output_path = os.path.join(checkpoint_dir, f"{prefix}embeddings.png")
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Embeddings plot saved to {output_path}")

    
def set_all_seeds_for_reproducibility(seed = 42):
    try:
        import random
        random.seed(seed)
    except Exception as e:
        print(f"Error setting random seed: {e}")
    
    try:
        import numpy as np
        np.random.seed(seed)
    except Exception as e:
        print(f"Error setting numpy seed: {e}")
        
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception as e:
        print(f"Error setting torch seed: {e}")
        
    print(f"Seeds set for reproducibility: {seed}")
    