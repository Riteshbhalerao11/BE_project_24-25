#!/bin/bash

nvidia-smi

num_epochs=$1
model=$2
val_num_samples=$3
master_port=$4

if [[ "$val_num_samples" == "1k" ]]; then
    num_val=1000
elif [[ "$val_num_samples" == "10k" ]]; then
    num_val=9253
else
    num_val=$val_num_samples
fi

finetune_script="finetune.py"

if [[ "$model" == "Flan-T5-XXL" ]]; then
    base_model="google/flan-t5-xxl"
    lora_target_modules='[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    prompt_template_name="alpaca"

elif [[ "$model" == "Llama-2-13B-chat" ]]; then
    base_model="meta-llama/Llama-2-13b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"

elif [[ "$model" == "Llama-2-7B-chat" ]]; then
    base_model="meta-llama/Llama-2-7b-chat-hf"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="alpaca"

elif [[ "$model" == "SmolLM-1.7B-Instruct" ]]; then
    base_model="HuggingFaceTB/SmolLM-1.7B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="chatML"
    echo "$base_model"

elif [[ "$model" == "SmolLM2-1.7B-Instruct" ]]; then
    base_model="HuggingFaceTB/SmolLM2-1.7B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="chatML"
    echo "$base_model"

elif [[ "$model" == "SmolLM2-360M-Instruct" ]]; then
    base_model="HuggingFaceTB/SmolLM2-360M-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="chatML"
    echo "$base_model"

elif [[ "$model" == "Llama-3.2-1B-Instruct" ]]; then
    base_model="meta-llama/Llama-3.2-1B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="Llama"
    echo "$base_model"

elif [[ "$model" == "Llama-3.2-3B-Instruct" ]]; then
    base_model="meta-llama/Llama-3.2-3B-Instruct"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="Llama"
    echo "$base_model"

elif [[ "$model" == "DeepSeek-R1-Distill-Qwen-1.5B" ]]; then
    base_model="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="qwen"
    finetune_script="finetune_reason.py"
    echo "$base_model"

elif [[ "$model" == "Mistral-7B-Instruct-v0.2" ]]; then
    base_model="mistralai/Mistral-7B-Instruct-v0.2"
    lora_target_modules='[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    prompt_template_name="mistral"

elif [[ "$model" == "Flan-T5-XL" ]]; then
    base_model="google/flan-t5-xl"
    lora_target_modules='[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    prompt_template_name="alpaca"

elif [[ "$model" == "Phi-2" ]]; then
    base_model="microsoft/phi-2"
    lora_target_modules='[Wqkv, out_proj, fc1, fc2, linear]'
    prompt_template_name="alpaca"

else
    base_model=""
    lora_target_modules=""
    prompt_template_name=""
fi

# Uncomment below to use multi-GPU training
# echo "$master_port"
# export CUDA_VISIBLE_DEVICES="0,1"
# accelerate launch --main_process_port $master_port finetune.py \

deepspeed "$finetune_script" \
    --deepspeed deepspeed_config.json \
    --base_model "$base_model" \
    --output_dir "./" \
    --batch_size 64 \
    --micro_batch_size 1 \
    --num_epochs "$num_epochs" \
    --cutoff_len 4000 \
    --val_set_size "$num_val" \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.10 \
    --lora_target_modules "$lora_target_modules" \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name "$prompt_template_name" \
    --lr_scheduler 'cosine' \
    --optim "adamw_torch" \
    --warmup_ratio 0.05
