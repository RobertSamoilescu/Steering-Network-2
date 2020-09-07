#!/bin/bash

BATCH_SIZE=$1
VIS_INT=$2
LOG_INT=$3
DATASET_DIR=$4
MODEL=$5

echo $MODEL" + RGB + SPEED"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_rgb \
	--use_speed \
	--use_balance \
	--dataset_dir $DATASET_DIR \
	--step_size 1000\
	--num_epochs 1000\
	--optimizer rmsprop\
	--weight_decay 0.0001 \
