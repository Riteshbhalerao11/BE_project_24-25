import os
import sys
import time
import pandas as pd

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import json

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
    prompt_template: str = "mistral"

):
    base_model = base_model or os.environ.get("BASE_MODEL", "")
    prompter = Prompter(prompt_template)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if device == "cuda":

        if base_model != 'microsoft/phi-2':
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_8bit=load_8bit,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                attn_implementation='flash_attention_2',
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
        tokenizer.padding_side = 'left'
        model.config.eos_token_id = tokenizer.eos_token_id
    else:
        tokenizer.pad_token_id = model.config.eos_token_id
        tokenizer.padding_side = 'left'

    pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer = tokenizer, 
        torch_dtype=torch.float16, 
        device_map="auto",
    )

    return prompter, tokenizer, pipe

prompter, tokenizer, pipe = main(False, False, "meta-llama/Llama-2-7b-chat-hf", "", "alpaca")

def response(instruction, input, options: str = "", output_data_path: str = "run_model/testing.json"):
    instructions = [instruction]
    inputs = [input]

    results = []
    max_batch_size = 1
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

        print(results)
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
    instruction = "Given the title, description, feature, price, and brand of a product and a set of target attributes, extract the value of each target attribute from the product information. Output the extracted value and the corresponding source (e.g., title or feature) denoting where the value is extracted."
    input = """
{
  "product title": "Digitek GoCAM DAC-002 5K 30fps 24MP WiFi Ultra HD Sports Action Camera with 2\" HD Screen & External MIC Support 100 feet Waterproof (with Waterproof case) 2 x 1350mAh Battery",
  "product description": "Digitek DAC-002 GoCAM has features you'll love exploring and Native 4K 60FPS crystal clear videos that will inspire the advanced photographer in you! Comes with dual screens feature. You can capture selfies in any extreme environment with to an intuitive 2″ LCD Touch Screen on the rare. Taking 6-Axis Gyro EIS Stabilization to a new level of stabilization, Super smooth brings you the capability to make crisp, shake-free, and butter-smooth image steadiness in the video like the camera is riding on its own rails. Water resistant to depth of up to 30 meters to meet most of the underwater sports records (with waterproof case) & With Wi-Fi support switch to the much faster on-demand to speed up file transfers and other app-based functions like low-latency image previews. With 128GB U3 Class+ 10 memory card(not included) for Capture crisp, pro-quality photos with 12MP interpolated to 24MP clarity. Digitek GoCAM DAC-001 can automatically pick all the best image processing for you, so it’s super easy to nail the shot. Shoot stunning video with up to 4K resolution, perfect for maintaining serious detail even when zooming in With 2nos of 1350mAh battery for long time operation, this camera almost covers all the functions that you need or you can imagine. For example, loop recording, time lapse recording, slow motion, self-timer, burst photo, screen saver, upside down, white balance and so on",
  "product brand": "Digitek",
  "target attributes": "Video Capture Resolution",
  "product feature": "Photo Sensor Technology CMOS, Video Capture Resolution 5K, Maximum Focal Length 2 Millimeters, Maximum Aperture 2 f, Flash Memory Type SD, Micro SD, Video Capture Format AVI,Screen Size 2 Inches, Connectivity Technology Wi-Fi, Colour grey"
}
    """
    response(instruction, input)

