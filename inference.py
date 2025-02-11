import os
import sys
import time
import pandas as pd

import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
import json

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter
import gc
import pdb
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

def main(
    load_8bit: bool = False,
    use_lora: bool = True,
    base_model: str = "../llama30B_hf",
    lora_weights: str = "",
    prompt_template: str = "mistral",
    # data_path: str = "",
    task: str = "",
    setting: str = "",
    output_data_path: str = ""

):
    print(locals())
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    directory = os.path.dirname(output_data_path)

    if not os.path.exists(directory):
        os.makedirs(directory)
        
    if device == "cuda":

        if base_model != 'microsoft/phi-2':
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )

        if use_lora:
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.bfloat16,
            )

    if not load_8bit:
        model.bfloat16()

    model.eval()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    if not model.config.eos_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
    elif not model.config.pad_token_id:
        tokenizer.pad_token_id = model.config.eos_token_id
    
    tokenizer.padding_side = 'left'

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.float16, 
        device_map="auto",
    )

    ## load from google drive
    # data_list = json.load(open(data_path, 'r'))
    # instructions = [data["instruction"] for data in data_list]
    # inputs = [data["input"] for data in data_list]
    # options = [data["options"] if "options" in data else None for data in data_list]

    # load from huggingface
    dataset = load_dataset("NingLab/ECInstruct")["train"]
    instructions, inputs, options = [], [], []
    for data in dataset:
        if data["split"] == "test" and data["task"] == task and data["setting"] == setting:
            instructions.append(data["instruction"])
            inputs.append(data["input"])
            options.append(data["options"])

    results = []
    max_batch_size = 2
    for i in range(0, len(instructions), max_batch_size):
        instruction_batch = instructions[i:i + max_batch_size]
        input_batch = inputs[i:i + max_batch_size]
        options_batch = options[i:i + max_batch_size]
        print(f"Processing batch {i // max_batch_size + 1} of {len(instructions) // max_batch_size + 1}...")
        start_time = time.time()
    
        prompts = [prompter.generate_prompt(instruction, input, options) for instruction, input, options in zip(instruction_batch, input_batch, options_batch)]
        batch_results = evaluate(prompter, prompts, tokenizer, pipe, max_batch_size)
            
        results.extend(batch_results)
        print(f"Finished processing batch {i // max_batch_size + 1}. Time taken: {time.time() - start_time:.2f} seconds")
    
    with open(output_data_path, 'w') as f:
        json.dump(results, f)

def evaluate(prompter, prompts, tokenizer, pipe, batch_size):
    batch_outputs = []

    generation_output = pipe(
        prompts,
        do_sample=True,
        max_new_tokens=100,
        temperature=0.15,
        top_p=0.95,
        num_return_sequences=1,
        num_beams=1,
        pad_token_id=tokenizer.eos_token_id,
        batch_size=batch_size,
    )

    for i in range(len(generation_output)):    
        resp = prompter.get_response(generation_output[i][0]['generated_text'])
        batch_outputs.append(resp)

    return batch_outputs


if __name__ == "__main__":
    torch.cuda.empty_cache()
    fire.Fire(main)

