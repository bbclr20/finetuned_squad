#!/bin/sh
conda activate transformers

export SQUAD_DIR=../squad/squad2

python run_squad.py \
    --model_type distilbert \
    --model_name_or_path distilbert-base-uncased \
    --do_train \
    --do_eval \
    --version_2_with_negative \
    --train_file ${SQUAD_DIR}/train-v2.0.json \
    --predict_file ${SQUAD_DIR}/dev-v2.0.json \
    --learning_rate 3e-5 \
    --num_train_epochs 1 \
    --max_seq_length 384 \
    --doc_stride 128 \
    --output_dir ./finetuned_squad/ \
    --per_gpu_eval_batch_size=8  \
    --per_gpu_train_batch_size=8   \
    --save_steps 5000