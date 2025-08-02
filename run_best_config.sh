#!/bin/bash

# Script to run the best configuration for the BDHPD paper

# Set the CUDA device
cuda_device=3

# Set the configuration file path
config_file=configs/config.yaml

# Set the checkpoint directory
checkpoint_dir="ckpt_dir"

# Set the configuration parameters based on the best model from the ablation study
wavelets=true
contrastive_loss=true
adain_layers=true
conv_bottleneck=true
balance_dataloaders=false
freeze_ssl=false

# Log file for the run
log_file="training_log_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a ${log_file})
exec 2> >(cat >&2)

echo "Starting training with best configuration at $(date)" >> ${log_file}
echo "Configuration: wavelets=${wavelets}, contrastive_loss=${contrastive_loss}, adain_layers=${adain_layers}, conv_bottleneck=${conv_bottleneck}" >> ${log_file}

# Run the training with the best configuration
CUDA_VISIBLE_DEVICES=${cuda_device} python train.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --training.balance_dataloaders=${balance_dataloaders} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck}

echo "Training process finished at $(date)" >> ${log_file}

# Run the testing with the same configuration
echo "Starting testing with the best model at $(date)" >> ${log_file}
CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck}

echo "Testing process finished at $(date)" >> ${log_file}
echo "All processes completed successfully."
