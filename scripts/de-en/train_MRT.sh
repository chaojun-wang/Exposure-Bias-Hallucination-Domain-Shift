#!/usr/bin/env sh
# Distributed under MIT license

script_dir=`dirname $0`
main_dir=$script_dir/../
data_dir=$main_dir/prep # date directory
working_dir=$main_dir/$1

mkdir $working_dir

# variables (toolkits; source and target language)
. $main_dir/vars

# TensorFlow devices; change this to control the GPUs used by Nematus.
# It should be a list of GPU identifiers. For example, '1' or '0,1,3'
devices=$2

CUDA_VISIBLE_DEVICES=$devices python3 $nematus_home/nematus/train.py \
    --source_dataset $data_dir/train.de-en.bpe.$src \
    --target_dataset $data_dir/train.de-en.bpe.$tgt \
    --dictionaries $data_dir/train.de-en.bpe.both.json \
                   $data_dir/train.de-en.bpe.both.json \
    --save_freq 100 \
    --model $working_dir/model \
    --reload $main_dir/$3/model.best-valid-script \
    --model_type transformer \
    --embedding_size 512 \
    --state_size 512 \
    --tie_encoder_decoder_embeddings \
    --tie_decoder_embeddings \
    --loss_function MRT \
    --label_smoothing 0 \
    --optimizer adam \
    --adam_beta1 0.9 \
    --adam_beta2 0.98 \
    --adam_epsilon 1e-09 \
    --learning_schedule constant \
    --learning_rate 0.00001 \
    --maxlen 100 \
    --batch_size 10 \
    --token_batch_size 0 \
    --valid_source_dataset $data_dir/dev.de-en.bpe.$src \
    --valid_bleu_source_dataset $data_dir/test.de-en.bpe.cat.$src \
    --valid_target_dataset $data_dir/dev.de-en.bpe.$tgt \
    --valid_batch_size 100 \
    --valid_token_batch_size 4096 \
    --valid_freq 50 \
    --valid_script $script_dir/validate_f.sh \
    --disp_freq 10 \
    --sample_freq 0 \
    --beam_freq 0 \
    --beam_size 4 \
    --translation_maxlen 100 \
    --normalization_alpha 1 \
    --transformer_enc_depth 6 \
    --transformer_dec_depth 6 \
    --patience 9999999 \
    --clip_c 1 \
    --transformer_dropout_embeddings 0.3 \
    --transformer_dropout_residual 0.3 \
    --transformer_dropout_relu 0.3 \
    --transformer_dropout_attn 0.3 \
    --sample_way randomly_sample \
    --samplesN 4 \
    --mrt_alpha 0.005 \
    --max_tokens_per_device 99999 \
    --max_sentences_of_sampling 99999 \
    --finish_after $4
    
