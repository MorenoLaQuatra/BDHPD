#!/bin/bash

# Script to run only the testing phase for the BDHPD model

# Set the CUDA device
cuda_device=1

# Set the configuration file path
config_file=configs/config.yaml

# Set the checkpoint directory (as specified)
checkpoint_dir="/mnt/disk3/mlaquatra/BDHPD_model/"

# Set the configuration parameters
wavelets=true
contrastive_loss=true
adain_layers=true
conv_bottleneck=true
freeze_ssl=false

# Log file for the run
log_file="testing_log_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a ${log_file})
exec 2> >(cat >&2)

echo "Starting testing with the best model at $(date)" >> ${log_file}
CUDA_VISIBLE_DEVICES=${cuda_device} python test.py --config ${config_file} \
    --training.checkpoint_dir=${checkpoint_dir} \
    --data.wavelets=${wavelets} \
    --model.freeze_ssl=${freeze_ssl} \
    --training.contrastive_loss.active=${contrastive_loss} \
    --model.use_adain_layers=${adain_layers} \
    --model.use_conv_bottleneck_layer=${conv_bottleneck}

echo "Testing process finished at $(date)" >> ${log_file}
echo "Testing completed successfully."
