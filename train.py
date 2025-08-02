import comet_ml
import torch
# functional
import torch.nn.functional as F
import numpy as np

from yaml_config_manager import load_config
from tqdm import tqdm
import os

from data_classes.ewadb_dataset import EWADBDataset
from model_classes.audio_classification_model import AudioClassificationModel
from miners.cl_miner import CLMiner

from additional_classes.checkpoint_manager import CheckpointManager
from list_dataloaders import ListDataLoaders

from utils import compute_metrics, get_dataset, create_optimizer_and_scheduler
from utils import get_device, create_model, get_experiment
from utils import get_classification_loss, resume_from_checkpoint_if_needed
from utils import save_results_file, save_confusion_matrix, get_single_dataloader
from utils import set_all_seeds_for_reproducibility, plot_embeddings, plot_embeddings_3d
from matplotlib import pyplot as plt

from pytorch_metric_learning import losses

def compute_classification_loss(config, criterion, batch, outputs):
    if config.model.num_classes == 2:
        logits = outputs["logits"].squeeze(1)
        targets = batch["labels"].float()
    else:
        logits = outputs["logits"]
        targets = batch["labels"]
    classification_loss = criterion(logits, targets)
    return classification_loss

def compute_contrastive_loss(config, criterion, batch, outputs, miner):
    embeddings = outputs["embeddings"]
    labels = batch["labels"]
    if miner is None:
        contrastive_loss = criterion(embeddings, labels)
    else:
        miner_output = miner.mine(embeddings, labels, batch["sample_type"], batch["domain_labels"])
        contrastive_loss = criterion(embeddings, labels, miner_output)
        
    # multiplier
    # print config
    print(f"Config: {config.training.contrastive_loss}")
    contrastive_loss *= config.training.contrastive_loss.multiplier
    return contrastive_loss

def train_one_epoch(config, model, dataloader, optimizer, scheduler, device, criterions, epoch, experiment, miner=None):
    model.train()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    
    p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}", leave=False)
    for i, batch in p_bar:
        
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        
        optimizer.zero_grad()
        outputs = model(batch)
        
        postfix_dict = {}
        loss = 0.0
        
        if config.training.cross_entropy_loss.active:
            classification_loss = compute_classification_loss(config, criterions["classification"], batch, outputs)
            postfix_dict = {"CLF-L": classification_loss.item()}
        else: classification_loss = None
        
        if config.training.contrastive_loss.active:
            contrastive_loss = compute_contrastive_loss(config, criterions["contrastive"], batch, outputs, miner)
            postfix_dict["CTR-L"] = contrastive_loss.item()
        else: contrastive_loss = None
        
        if classification_loss is not None: loss += classification_loss
        if contrastive_loss is not None: loss += contrastive_loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss += loss.item()
        if config.model.num_classes == 2:
            logits = outputs["logits"].squeeze(1)
            
            # apply sigmoid to binary classification
            current_predictions = torch.sigmoid(logits).detach().cpu().numpy()
            current_predictions = np.where(current_predictions > 0.5, 1, 0)
        else:
            # apply softmax to multiclass classification
            current_predictions = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
        
        all_labels.extend(batch["labels"].cpu().numpy())
        all_predictions.extend(current_predictions)
            
        p_bar.set_postfix(postfix_dict)
    
    metrics = compute_metrics(all_labels, all_predictions, is_binary_classification=config.model.num_classes == 2)
    metrics["loss"] = running_loss / len(dataloader)
    return metrics

def evaluate_one_epoch(config, model, dataloader, device, criterions, epoch, experiment, return_embeddings=False):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_predictions = []
    all_embeddings = []
    all_sample_types = []
    
    with torch.no_grad():
        p_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Validation {epoch}", leave=False)
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
                # apply sigmoid to binary classification
                current_predictions = torch.sigmoid(logits).detach().cpu().numpy()
                current_predictions = np.where(current_predictions > 0.5, 1, 0)
            else:
                # apply softmax to multiclass classification
                current_predictions = torch.softmax(logits, dim=-1).argmax(dim=-1).cpu().numpy()
            
            all_labels.extend(batch["labels"].cpu().numpy())
            all_predictions.extend(current_predictions)
            all_embeddings.extend(outputs["embeddings"].cpu().numpy())
            all_sample_types.extend(batch["sample_type"].cpu().numpy())
        
            p_bar.set_postfix({"loss": running_loss / (i + 1)})
            
    metrics = compute_metrics(all_labels, all_predictions, is_binary_classification=config.model.num_classes == 2)
    metrics["loss"] = running_loss / len(dataloader)
    if return_embeddings:
        return metrics, all_embeddings, all_labels, all_sample_types
    return metrics
    


def main(config):
    
    # load train, validation, and test datasets
    # train_dataset = get_dataset(config, "train")
    # test_dataset = get_dataset(config, "test")
    # validation_dataset = get_dataset(config, "validation")
    
    set_all_seeds_for_reproducibility()
    
    if config.ewadb.active:
        train_ewadb = get_dataset(config, "train", "ewadb", domain_id=0)
        validation_ewadb = get_dataset(config, "validation", "ewadb", domain_id=0)
    else:
        train_ewadb = None
        validation_ewadb = None
        
    if config.pc_gita.active:
        train_pcgita = get_dataset(config, "train", "pc_gita", domain_id=1)
        validation_pcgita = get_dataset(config, "validation", "pc_gita", domain_id=1)
    
    test_ewadb = get_dataset(config, "test", "ewadb", domain_id=0)
    test_pcgita = get_dataset(config, "test", "pc_gita", domain_id=1)
    
    # # merge train datasets using torch.utils.data.ConcatDataset
    if config.ewadb.active and config.pc_gita.active:
        train_dataset = torch.utils.data.ConcatDataset([train_ewadb, train_pcgita])
    elif config.ewadb.active:
        train_dataset = train_ewadb
    elif config.pc_gita.active:
        train_dataset = train_pcgita
    else:
        raise ValueError(f"At least one of the datasets should be active: pc_gita: {config.pc_gita.active}, ewadb: {config.ewadb.active}")
    
    # create dataloader
    train_dl = get_single_dataloader(config, train_dataset, "train", balance_dataloader=config.training.balance_dataloaders)
    if config.ewadb.active and config.pc_gita.active:
        val_dl_ewadb = get_single_dataloader(config, validation_ewadb, "validation")
        val_dl_pcgita = get_single_dataloader(config, validation_pcgita, "validation")
    elif config.ewadb.active:
        val_dl_ewadb = get_single_dataloader(config, validation_ewadb, "validation")
        val_dl_pcgita = None
    elif config.pc_gita.active:
        val_dl_ewadb = None
        val_dl_pcgita = get_single_dataloader(config, validation_pcgita, "validation")
    
    
    test_dl_ewadb = get_single_dataloader(config, test_ewadb, "test")
    test_dl_pcgita = get_single_dataloader(config, test_pcgita, "test")
    
    # train_dl = ListDataLoaders([train_dl_ewadb, train_dl_pcgita], weight_by_num_samples=True)
    
    print ("Datasets loaded successfully")
    if config.ewadb.active:  print ("Train EWADB dataset length: ", len(train_ewadb))
    if config.pc_gita.active: print ("Train PCGITA dataset length: ", len(train_pcgita))
    
    if config.ewadb.active:  print ("Validation EWADB dataset length: ", len(validation_ewadb))
    if config.pc_gita.active: print ("Validation PCGITA dataset length: ", len(validation_pcgita))
    
    print ("Test EWADB dataset length: ", len(test_ewadb))
    print ("Test PCGITA dataset length: ", len(test_pcgita))
    
    # set number of domains
    config.model.num_domains = 2
    
    experiment = get_experiment(config)
    device = get_device(config)
    print(f"\t\tUsing device: {device}")
    model = create_model(config, device)
    len_for_opt_and_sched = 0
    if config.ewadb.active: len_for_opt_and_sched += len(train_ewadb)
    if config.pc_gita.active: len_for_opt_and_sched += len(train_pcgita)
    optimizer, scheduler = create_optimizer_and_scheduler(model, config, len_for_opt_and_sched)
    
    checkpoint_manager = CheckpointManager(
        checkpoint_dir = config.training.checkpoint_dir,
        model = model,
        optimizer = optimizer,
        scheduler = scheduler,
        device = device,
        lower_is_better = config.validation.metric_lower_is_better
    )
    
    criterions = {}
    criterions["classification"] = get_classification_loss(config.model.num_classes)
    # contrastive learning loss
    if config.training.contrastive_loss.active:
        criterions["contrastive"] = losses.ContrastiveLoss()
        miner = CLMiner()
    else: 
        miner = None
        

    
    # resume if needed
    start_epoch, best_metric = resume_from_checkpoint_if_needed(config, checkpoint_manager)
    
    # train loop
    max_epochs_without_improvement = 5
    current_epochs_without_improvement = 0
    for epoch in range(start_epoch, config.training.num_epochs):
        print(f"Epoch {epoch} started")
        
        train_metrics = train_one_epoch(config, model, train_dl, optimizer, scheduler, device, criterions, epoch, experiment, miner)
        
        if val_dl_ewadb is not None: val_metrics_ewadb = evaluate_one_epoch(config, model, val_dl_ewadb, device, criterions, epoch, experiment)
        else: val_metrics_ewadb = None
        
        if val_dl_pcgita is not None: val_metrics_pcgita = evaluate_one_epoch(config, model, val_dl_pcgita, device, criterions, epoch, experiment)
        else: val_metrics_pcgita = None
        
        print(f"Epoch {epoch} Train Loss: {train_metrics['loss']}")
        print(f"Epoch {epoch} Train Metrics: {train_metrics}")
        
        if val_metrics_ewadb is not None:
            print(f"[EWADB] Epoch {epoch} metrics:")
            for m in val_metrics_ewadb: print(f"Val {m}: {val_metrics_ewadb[m]}")
            for m in val_metrics_ewadb: experiment.log(f"val_ewadb_{m}", val_metrics_ewadb[m])
            
        if val_metrics_pcgita is not None:
            print(f"[PCGITA] Epoch {epoch} metrics:")
            for m in val_metrics_pcgita: print(f"Val {m}: {val_metrics_pcgita[m]}")
            for m in val_metrics_pcgita: experiment.log(f"val_pcgita_{m}", val_metrics_pcgita[m])
        
        # add epoch_ as prefix to all metrics
        epoch_metrics = { f"epoch_{k}": v for k, v in train_metrics.items() }
        
        if "+" in config.training.validation.metric:
            m_to_be_used = config.training.validation.metric.split("+")
        else:
            m_to_be_used = [config.training.validation.metric]
            
        current_metric = []
        if val_metrics_ewadb is not None: 
            for m in m_to_be_used: current_metric.append(val_metrics_ewadb[m])
        if val_metrics_pcgita is not None: 
            for m in m_to_be_used: current_metric.append(val_metrics_pcgita[m])
        
        current_metric = sum(current_metric) / len(current_metric)
        
        # current metric is the average of the metrics from both datasets
        print(f"Current Metric: {current_metric}")
        print(f"Previous best metric: {checkpoint_manager.get_current_best_metric()}")
        current_is_best = checkpoint_manager.save_checkpoint(epoch, current_metric)
        if current_is_best:
            current_epochs_without_improvement = 0
        else:
            current_epochs_without_improvement += 1
            print(f"Current epochs without improvement: {current_epochs_without_improvement}")
            if current_epochs_without_improvement >= max_epochs_without_improvement:
                print(f"Early stopping at epoch {epoch} - no improvement for {max_epochs_without_improvement} epochs")
                break
    
    # load best model
    checkpoint_manager.load_best_model()
    
    # separate evaluation for test datasets
    test_metrics_ewadb, embeddings_ewadb, labels_ewadb, sample_types_ewadb = evaluate_one_epoch(config, model, test_dl_ewadb, device, criterions, "test", experiment, return_embeddings=True)
    test_metrics_pcgita, embedding_pcgita, labels_pcgita, sample_types_pcgita = evaluate_one_epoch(config, model, test_dl_pcgita, device, criterions, "test", experiment, return_embeddings=True)
    
    print(f"[EWADB] Test Metrics")
    for m in test_metrics_ewadb: print(f"Test {m}: {test_metrics_ewadb[m]}")
    
    print(f"[PCGITA] Test Metrics")
    for m in test_metrics_pcgita: print(f"Test {m}: {test_metrics_pcgita[m]}")
    
    # separately store results for both test datasets
    save_results_file(config.training.checkpoint_dir, test_metrics_ewadb, prefix="ewadb_")
    save_results_file(config.training.checkpoint_dir, test_metrics_pcgita, prefix="pcgita_")
    
    # separately store confusion matrices for both test datasets
    save_confusion_matrix(config.training.checkpoint_dir, test_metrics_ewadb["confusion_matrix"], prefix="ewadb_")
    save_confusion_matrix(config.training.checkpoint_dir, test_metrics_pcgita["confusion_matrix"], prefix="pcgita_")
    
    # separately plot embeddings for both test datasets
    plot_embeddings(config.training.checkpoint_dir, embeddings_ewadb, labels_ewadb, sample_types_ewadb, prefix="ewadb_")
    plot_embeddings(config.training.checkpoint_dir, embedding_pcgita, labels_pcgita, sample_types_pcgita, prefix="pcgita_")
    
    plot_embeddings_3d(config.training.checkpoint_dir, embeddings_ewadb, labels_ewadb, sample_types_ewadb, prefix="ewadb_")
    plot_embeddings_3d(config.training.checkpoint_dir, embedding_pcgita, labels_pcgita, sample_types_pcgita, prefix="pcgita_")

    
if __name__ == "__main__":
    config = load_config()
    main(config)