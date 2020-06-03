#!/bin/bash

BATCH_SIZE=$1
VIS_INT=$2
LOG_INT=$3
DATASET_DIR=$4
MODEL=$5

echo $MODEL" + RGB + STACKED"
python3 test.py \
	--model $MODEL\
	--batch_size $BATCH_SIZE \
	--use_rgb \
	--use_stacked \
	--use_balance \
	--dataset_dir $DATASET_DIR\