#!/bin/bash


# BERT models
berts=( "google-bert/bert-base-multilingual-cased" );
# berts=( "FacebookAI/xlm-roberta-base" );
# berts=( "uitnlp/visobert" );
# berts=( "vinai/phobert-base" );
# berts=( "uitnlp/CafeBERT" );

# datasets
datadir="data/sent_graphs"
datasets=( "vietnam/head_final" );
datasets_slur="vietnam"

model_slur="mbert"
# model_slur="xlm-r"
# model_slur="visobert"
# model_slur="phobert"
# model_slur="cafebert"

mkdir -p $datadir/$datasets_slur/$model_slur

for ((i=0;i<${#berts[@]};++i)); do
  model="${berts[i]}"
  for t in train dev test; do
    indata="$datadir"/"${datasets[i]}"/"$t".conllu
    # outfile="$datadir"/"${datasets[i]}"/"$t"_bert.hdf5
    outfile="$datadir"/"${datasets_slur}"/"$model_slur"/"$t"_bert.hdf5
    printf "Using %s for %s\n" "$model" "$indata";
    printf "Saving to %s\n" "$outfile"
    python3 context_embed.py --model $model --indata $indata --outdata $outfile;
  done;
  echo
done;




