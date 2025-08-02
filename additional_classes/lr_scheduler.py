import torch
from torch.optim import Optimizer
import math
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler

class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(self, optimizer: Optimizer, total_steps: int, decay_type: str = 'linear',
                 warmup_steps: int = None, warmup_ratio: float = None, last_epoch: int = -1):
        """
        Custom learning rate scheduler with optional warmup and different decay options.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_steps (int): Total number of training steps.
            decay_type (str): Decay type to use ('linear', 'cosine', 'exponential').
            warmup_steps (int, optional): Number of warmup steps. Default is None.
            warmup_ratio (float, optional): Warmup ratio (used if warmup_steps is None). Default is None.
            last_epoch (int, optional): The index of last epoch. Default is -1.

        Raises:
            ValueError: If both warmup_steps and warmup_ratio are set or if decay_type is not supported.
        """
        self.total_steps = total_steps
        self.decay_type = decay_type.lower()

        if warmup_steps is not None and warmup_ratio is not None:
            raise ValueError("Only one of 'warmup_steps' or 'warmup_ratio' should be set.")
        
        if warmup_ratio is not None:
            self.warmup_steps = int(total_steps * warmup_ratio)
        else:
            self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_step = self.last_epoch + 1

        if self.warmup_steps and current_step < self.warmup_steps:
            return [base_lr * current_step / self.warmup_steps for base_lr in self.base_lrs]

        if self.decay_type == 'linear':
            return [base_lr * (1 - (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps))
                    for base_lr in self.base_lrs]
        elif self.decay_type == 'cosine':
            return [base_lr * 0.5 * (1 + math.cos(math.pi * (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
                    for base_lr in self.base_lrs]
        elif self.decay_type == 'exponential':
            return [base_lr * (0.9 ** ((current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)))
                    for base_lr in self.base_lrs]
        else:
            raise ValueError(f"Decay type '{self.decay_type}' is not supported.")

    def _create_lambda(self):
        if self.decay_type == 'linear':
            return lambda step: 1 - step / (self.total_steps - self.warmup_steps)
        elif self.decay_type == 'cosine':
            return lambda step: 0.5 * (1 + math.cos(math.pi * step / (self.total_steps - self.warmup_steps)))
        elif self.decay_type == 'exponential':
            return lambda step: 0.9 ** (step / (self.total_steps - self.warmup_steps))
        else:
            raise ValueError(f"Decay type '{self.decay_type}' is not supported.")

if __name__ == "__main__":
    # Example of usage
    optimizer = torch.optim.Adam([torch.zeros(3, requires_grad=True)], lr=1e-3)
    total_steps = 1000

    # Initialize schedulers for each decay type
    schedulers = {
        'linear': LRSchedulerWithWarmup(optimizer, total_steps=total_steps, decay_type='linear', warmup_ratio=0.1),
        'cosine': LRSchedulerWithWarmup(optimizer, total_steps=total_steps, decay_type='cosine', warmup_steps=100),
        'exponential': LRSchedulerWithWarmup(optimizer, total_steps=total_steps, decay_type='exponential', warmup_steps=100)
    }

    # Plot the learning rate curves
    plt.figure(figsize=(10, 6))
    for decay_type, scheduler in schedulers.items():
        lrs = []
        for step in range(total_steps):
            optimizer.step()
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])

        plt.plot(lrs, label=decay_type)

    plt.xlabel("Training Steps")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedules")
    plt.legend()
    plt.grid(True)

    # Save plot as PNG
    plt.savefig("learning_rate_schedules.png")
    plt.show()
