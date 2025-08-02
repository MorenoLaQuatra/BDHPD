import os
import torch

class CheckpointManager:
    """
    A class to manage saving and loading of model checkpoints, including optimizer, scheduler, and tracking metrics.

    Attributes:
        checkpoint_dir (str): Directory where checkpoints will be saved.
        model (torch.nn.Module): The model to be saved or loaded.
        optimizer (torch.optim.Optimizer): The optimizer to be saved or loaded.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to be saved or loaded.
        device (torch.device): The device on which the model is loaded.
        best_metric (float): The best metric achieved so far.
        lower_is_better (bool): Flag to determine if lower metric values are better.
    """

    def __init__(self, checkpoint_dir, model=None, optimizer=None, scheduler=None, device=None, lower_is_better=False):
        """
        Initialize the CheckpointManager.

        Args:
            checkpoint_dir (str): Directory to save and load checkpoints.
            model (torch.nn.Module): The model to manage.
            optimizer (torch.optim.Optimizer): The optimizer to manage.
            scheduler (torch.optim.lr_scheduler._LRScheduler): The scheduler to manage.
            device (torch.device): The device for loading the checkpoint.
            lower_is_better (bool): Set to True if a lower metric indicates better performance.
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_metric = None
        self.lower_is_better = lower_is_better

    def save_checkpoint(self, epoch, current_metric, is_best=False):
        """
        Save the model, optimizer, and scheduler state to a checkpoint.

        Args:
            epoch (int): The current epoch number.
            current_metric (float): The metric value for the current epoch.
            is_best (bool): Flag to save the current checkpoint as the best model.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric
        }
        
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        current_is_best = False
        # Update the best metric and save the best model if needed
        if self.best_metric is None or (self.lower_is_better and current_metric < self.best_metric) or (not self.lower_is_better and current_metric > self.best_metric):
            self.best_metric = current_metric
            best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
            torch.save(checkpoint, best_checkpoint_path)
            print(f"Best model saved at {best_checkpoint_path}")
            current_is_best = True
            
        return current_is_best

    def load_checkpoint(self, checkpoint_path, load_optimizer_scheduler=False):
        """
        Load the model, optimizer, and scheduler state from a checkpoint.

        Args:
            checkpoint_path (str): The path to the checkpoint file.
            load_optimizer_scheduler (bool): Whether to load the optimizer and scheduler states.

        Returns:
            int: The epoch number of the loaded checkpoint.
            float: The best metric value saved in the checkpoint.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, torch.nn.DataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer_scheduler:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.best_metric = checkpoint.get('best_metric', None)
        print(f"Checkpoint loaded from {checkpoint_path}")
        
        return checkpoint['epoch'], self.best_metric

    def list_checkpoints(self):
        """
        List all checkpoints in the checkpoint directory.

        Returns:
            list: A sorted list of checkpoint file names.
        """
        files = os.listdir(self.checkpoint_dir)
        checkpoints = [f for f in files if f.endswith('.pt')]
        return sorted(checkpoints)
    
    def get_current_best_metric(self):
        """
        Get the best metric value saved in the checkpoint.

        Returns:
            float: The best metric value saved in the checkpoint.
        """
        return self.best_metric

    def get_latest_checkpoint(self):
        """
        Get the path to the latest checkpoint.

        Returns:
            str: The path to the latest checkpoint or None if no checkpoints exist.
        """
        checkpoints = self.list_checkpoints()
        if not checkpoints:
            return None
        latest_checkpoint = checkpoints[-1]
        return os.path.join(self.checkpoint_dir, latest_checkpoint)

    def load_best_model(self):
        """
        Load the best model from the checkpoint directory.

        Returns:
            float: The best metric value saved in the checkpoint.
        """
        best_checkpoint_path = os.path.join(self.checkpoint_dir, "best_model.pt")
        if not os.path.exists(best_checkpoint_path):
            raise FileNotFoundError(f"No best model found at {best_checkpoint_path}")

        _, best_metric = self.load_checkpoint(best_checkpoint_path, load_optimizer_scheduler=False)
        return best_metric