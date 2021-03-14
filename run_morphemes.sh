#!/usr/bin/env bash
# Experiments on morphemes

DATA_DIR=./data/morpheme
MODEL_DIR=./model
MODEL=$MODEL_DIR/morphemes

if [ ! -d $MODEL_DIR ]; then
  mkdir $MODEL_DIR
fi
if [ ! -d $MODEL ]; then
  mkdir $MODEL
fi
#################################################################################

SUB_MODEL=wd_eg_mor_fm_sm

CUDA_VISIBLE_DEVICES=5 python3 main.py \
    --data $DATA_DIR \
    --vocab_size 60000 \
    --lr_decay 0.3 \
    --epochs 500 \
    --vec $DATA_DIR/gigaword_300d_jieba_unk.bin \
    --batch_size 64 \
    --cuda \
    --save $MODEL/$SUB_MODEL.pt \
    --init_embedding \
    --tied \
    --clip 5 \
    --init_dist uniform \
    --dropout 0.2 \
    --teacher_ratio 0.9 \
    --lr 0.001 \
    --nlayers 1 \
    --seed_feeding \
    --char \
    --use_wordvec \
    --use_eg \
    --use_morpheme \
    --use_formation \
    --use_scheme \
    --data_usage 100 \
    --log_interval 10 \
    >$MODEL/$SUB_MODEL.train.log