#!/bin/bash

export PATH="/sysapps/ubuntu-applications/miniconda/4.12.0/miniconda3/bin:$PATH"
cd ~/playground/fMRI_SSM

source activate 
conda activate fmamba

# NCCL_P2P_LEVEL=NVL torchrun --nnodes=1 --nproc_per_node=2 ~/playground/fMRI_SSM/main.py \
python ~/playground/fMRI_SSM/main.py \
  --model_name=FunctionalMamba \
  --dataset_name=HCP_dfnc \
  --image_format=2DT \
  --target_name=age \
  --task=classification \
  --num_classes=1 \
  --epoch=200 \
  --batch_size=64 \
  # --enable_mlflow \
  # --save_model \
  # --resume \
  # --save_model \
  # --distributed \