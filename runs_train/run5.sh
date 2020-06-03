#!/bin/bash

BATCH_SIZE=$1
VIS_INT=$2
LOG_INT=$3
DATASET_DIR=$4
MODEL=$5

echo $MODEL" + RGB + STACKED + DISP"
python3 train.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--vis_int $VIS_INT \
	--log_int $LOG_INT \
	--use_rgb \
	--use_stacked \
	--use_disp \
	--use_balance \
	--dataset_dir $DATASET_DIR\
