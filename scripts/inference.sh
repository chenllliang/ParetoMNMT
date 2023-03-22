#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


lang_pairs="en_XX-de_DE,en_XX-fr_XX,en_XX-zh_CN"
path_2_data=../databin
lang_list=./angs.txt
model=$1

source_lang=(en_XX)
out_dir=$1-inference-result
mkdir $out_dir



TGT=(de_DE fr_XX zh_CN)

export CUDA_VISIBLE_DEVICES=3

for i in $(seq 0 0); do

for j in $(seq 0 2); do

if [ ${source_lang[$i]} != ${TGT[$j]} ]

then

echo testing ${source_lang[$i]} - ${TGT[$j]}


fairseq-generate "$path_2_data" \
  --path $model \
  --task translation_multi_simple_epoch \
  --gen-subset valid \
  --source-lang "${source_lang[$i]}" \
  --target-lang "${TGT[$j]}" \
  --batch-size 100 --remove-bpe 'sentencepiece'\
  --encoder-langtok "tgt" \
  --beam 5 \
  --left-pad-source False \
  --lang-dict "$lang_list" \
  --lang-pairs "$lang_pairs" > $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5


cat $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5 | grep -P "^H"  |cut -f 3-  > $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp
cat $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5 | grep -P "^T"  |cut -f 2-  > $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref

fi

if [ ${TGT[$j]} == "zh_CN" ]
then
echo sacrebleu ${TGT[$j]},using chinese version
sacrebleu $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref --metrics {bleu,chrf} --tokenize zh < $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp
sacrebleu $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref --metrics {bleu,chrf} --tokenize zh < $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp > $out_dir/${source_lang[$i]}-${TGT[$j]}.beam5.sacrebleu

else
echo sacrebleu ${TGT[$j]}
sacrebleu $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref --metrics {bleu,chrf} < $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp
sacrebleu $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.ref --metrics {bleu,chrf} < $out_dir/${source_lang[$i]}-${TGT[$j]}.out.beam5.hyp > $out_dir/${source_lang[$i]}-${TGT[$j]}.beam5.sacrebleu
fi


done
done
