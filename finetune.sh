#!/bin/tcsh


nvidia-smi


set num_epochs = $1
set model = $2
set val_num_samples = $3
set master_port = $4

if ($val_num_samples == "1k") then
    set num_val = 1000
else if ($val_num_samples == "10k") then
    set num_val = 9253
else
    set num_val = $val_num_samples
endif 

set finetune_script = "finetune.py"

if ($model == "Flan-T5-XXL") then
    set base_model = "google/flan-t5-xxl"
    set lora_target_modules = '[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    set prompt_template_name = alpaca
else if ($model == "Llama-2-13B-chat") then
    set base_model = "meta-llama/Llama-2-13b-chat-hf"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = alpaca
else if ($model == "Llama-2-7B-chat") then
    set base_model = "meta-llama/Llama-2-7b-chat-hf"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = alpaca
    
else if ($model == "SmolLM-1.7B-Instruct") then
    set base_model = "HuggingFaceTB/SmolLM-1.7B-Instruct"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = chatML
    echo $base_model

else if ($model == "SmolLM2-1.7B-Instruct") then
    set base_model = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = chatML
    echo $base_model

else if ($model == "SmolLM2-360M-Instruct") then
    set base_model = "HuggingFaceTB/SmolLM2-360M-Instruct"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = chatML
    echo $base_model

else if ($model == "Llama-3.2-1B-Instruct") then
    set base_model = "meta-llama/Llama-3.2-1B-Instruct"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = Llama
    echo $base_model

else if ($model == "Llama-3.2-3B-Instruct") then
    set base_model = "meta-llama/Llama-3.2-3B-Instruct"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = Llama
    echo $base_model

else if ($model == "DeepSeek-R1-Distill-Qwen-1.5B") then
    set base_model = "/kaggle/working/DeepSeek-R1-Distill-Qwen-1.5B"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = qwen
    set finetune_script = "finetune_reason.py"
    echo $base_model

else if ($model == "Mistral-7B-Instruct-v0.2") then
    set base_model = "mistralai/Mistral-7B-Instruct-v0.2"
    set lora_target_modules = '[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head]'
    set prompt_template_name = mistral

else if ($model == "Flan-T5-XL") then
    set base_model = "google/flan-t5-xl"
    set lora_target_modules = '[q, k, v, o, wi_0, wi_1, wo, lm_head]'
    set prompt_template_name = alpaca
else if ($model == "Phi-2") then
    set base_model = "microsoft/phi-2"
    set lora_target_modules = '[Wqkv, out_proj, fc1, fc2, linear]'
    set prompt_template_name = alpaca
else
    set base_model = ""
    set lora_target_modules  = ""
    set prompt_template_name = ""
endif

#uncomment the following code to apply multi-gpu training
echo $master_port
setenv CUDA_VISIBLE_DEVICES "0,1,2,3"
accelerate launch --main_process_port $master_port $finetune_script\
    --base_model $base_model \
    --output_dir "./" \
    --batch_size 48 \
    --micro_batch_size 1 \
    --num_epochs $num_epochs \
    --cutoff_len 4000 \
    --val_set_size $num_val \
    --learning_rate 1e-4 \
    --lora_r 16 \
    --lora_alpha 16 \
    --lora_dropout 0.10 \
    --lora_target_modules "$lora_target_modules" \
    --train_on_inputs False \
    --add_eos_token False \
    --group_by_length False \
    --prompt_template_name $prompt_template_name \
    --lr_scheduler 'cosine' \
    --optim "adamw_torch" \
    --warmup_ratio 0.05 \
    --wandb_project "test" \
    --wandb_run_name "test" \
    --wandb_watch "true" \
    --wandb_log_model "false" \




# python $finetune_script \
#     --base_model $base_model \
#     --output_dir "./" \
#     --batch_size 64 \
#     --micro_batch_size 1 \
#     --num_epochs $num_epochs \
#     --cutoff_len 4000 \
#     --val_set_size $num_val \
#     --learning_rate 1e-4 \
#     --lora_r 16 \
#     --lora_alpha 16 \
#     --lora_dropout 0.10 \
#     --lora_target_modules "$lora_target_modules" \
#     --train_on_inputs False \
#     --add_eos_token False \
#     --group_by_length False \
#     --prompt_template_name $prompt_template_name \
#     --lr_scheduler 'cosine' \
#     --optim "adamw_torch" \
#     --warmup_ratio 0.05 \
#     --wandb_project "test" \
#     --wandb_run_name "test" \
#     --wandb_watch "true" \
#     --wandb_log_model "false" \

