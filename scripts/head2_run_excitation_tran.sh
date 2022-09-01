#!/bin/bash
CUDA_DEVICE_ORDER=PCI_BUS_ID
WS=`pwd`

n_seqs=1000
n_types=10
n_correlations=16
dataset=mhp-$(($n_seqs / 1000))K-$n_types
shared_args="--dataset $dataset"
n_splits=5  # if modified, remember to modify below as well!!!

LOCAL_RUN="xargs -L1 -P${n_splits} python"

if [ ! -d pkg ]; then
    echo "Please execute the script at the root project directory." && exit
elif [ $# == 0 ]; then
    echo "No argument provided."
    exit 1
fi


# preprocessing/data generation

if [[ $* == *all* ]] || [[ $* == *preprocess* ]]; then
    python preprocessing/generate_events_by_mhp.py \
        --n_seqs $n_seqs \
        --n_types $n_types \
        --n_correlations $n_correlations \
        --baseline 0.01 \
        --exp_decay 0.05  \
        --adj_spectral_radius 0.8 \
        --max_jumps 500 \
        --n_splits $n_splits
        --fit
fi

# training for each methods

if [[ $* == *all* ]] || [[ $* == *Tran* ]]; then
    printf "%s\n" "$WS/tasks/train_tran.py Tran $shared_args --tran_layer 1 --tran_head 2 --cuda --split_id "{0..4} | $LOCAL_RUN
    printf "%s\n" "$WS/tasks/train_tran.py Tran $shared_args --tran_layer 2 --tran_head 2 --cuda --split_id "{0..4} | $LOCAL_RUN
    printf "%s\n" "$WS/tasks/train_tran.py Tran $shared_args --tran_layer 3 --tran_head 2 --cuda --split_id "{0..4} | $LOCAL_RUN
fi
# python postprocessing/summarize_results.py $dataset
