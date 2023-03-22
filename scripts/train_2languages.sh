#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

echo pid:$$



lang_list=./langs.txt  # <path to a file which contains a list of languages separted by new lines>
lang_pairs=en_XX-fr_XX,en_XX-zh_CN #a list language pairs to train multilingual models, e.g. "en-fr,en-cs,fr-en,cs-en"
# pretrained can be an mBART pretrained model as well

#lang_pairs=en_XX-tr_TR
BIN=../databin
SAVE=../checkpoints

DATE=0322
export CUDA_VISIBLE_DEVICES=3

cat $0
mkdir ../logs/${DATE}
mkdir ../checkpoints/${DATE}

ratios=$1
echo "["$ratios"]" 
TASK=base_model_2d_10Mfr260kzh_$ratios

fairseq-train "${BIN}" \
  --save-dir ${SAVE}/${DATE}/${TASK} \
  --arch transformer \
  --encoder-layers 3 \
  --decoder-layers 3 \
  --share-all-embeddings \
  --task translation_multi_simple_epoch_dpl \
  --encoder-langtok "tgt" \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" \
  --best-checkpoint-metric ppl \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler inverse_sqrt --lr 3e-04 --warmup-updates 4000 --max-update 100000 \
  --save-interval 1 --save-interval-updates 20000 \
  --keep-interval-updates 10 \
  --no-epoch-checkpoints \
  --left-pad-source False \
  --skip-invalid-size-inputs-valid-test \
  --max-source-positions 500 --max-target-positions 500 \
  --seed 888 --log-format json --log-interval 100 \
  --max-tokens 20000 \
  --wandb-project ParetoMNMT \
  --fp16 \
  --log-file ../logs/${DATE}/$TASK.log \
  --log-task-valid-loss \
  --validate-interval-updates 5000 \
  --sampling-method "predefined" \
  --sampling-ratios "["$ratios"]" 
