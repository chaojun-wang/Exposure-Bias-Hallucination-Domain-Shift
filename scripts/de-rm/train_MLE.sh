#!/usr/bin/env sh
# Distributed under MIT license

script_dir=`dirname $0`
main_dir=$script_dir/../
data_dir=$main_dir/data/de-rm/law # date directory
working_dir=$main_dir/$1

mkdir $working_dir

# variables (toolkits; source and target language)
. $main_dir/vars

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
devices=$2


CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $data_dir/train.bpe.$src \
    --target_dataset $data_dir/train.bpe.$tgt \
    --dictionaries $data_dir/train.bpe.both.json \
                   $data_dir/train.bpe.both.json \
    --save_freq 30000 \
    --model $working_dir/model \
    --reload latest_checkpoint \
    --model_type transformer \
    --embedding_size 512 \
    --state_size 512 \
    --tie_encoder_decoder_embeddings \
    --tie_decoder_embeddings \
    --loss_function per-token-cross-entropy \
    --label_smoothing 0.1 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --learning_schedule transformer \
    --warmup_steps 6000 \
    --maxlen 100 \
    --batch_size 64 \
    --token_batch_size 4096 \
    --valid_source_dataset $data_dir/dev.bpe.$src \
    --valid_bleu_source_dataset $data_dir/dev.bpe.cat.$src \
    --valid_target_dataset $data_dir/dev.bpe.$tgt \
    --valid_batch_size 100 \
    --valid_token_batch_size 4096 \
    --valid_freq 500 \
    --valid_script $script_dir/validate.sh \
    --disp_freq 125 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 4 \
    --translation_maxlen 100 \
    --normalization_alpha 1 \
    --transformer_enc_depth 6 \
    --transformer_dec_depth 6 \
    --patience 32 \
    --clip_c 0 
    
