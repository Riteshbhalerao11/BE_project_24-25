from pathlib import Path
import re
import json
import os
pattern = r"\\boxed\{(.*?)\}"

def extract_boxed_escaped_string(text):
    match = re.search(pattern, text, re.DOTALL)
    if match:
        raw_string = match.group(1)
        return raw_string
    return None


def transform_path(path):
    # Get the parts of the path
    parts = list(path.parts)
    
    # Find the index of "qwen_reasoning_raw" in the parts
    for i, part in enumerate(parts):
        if part == "qwen_reasoning_raw":
            parts[i] = "qwen_reasoning"
            break
    
    # Reconstruct the path
    return Path(*parts)

all_files = list(Path("/content/BE_project_24-25/inference_results/qwen_reasoning_raw").rglob("*.*"))

for my_file in all_files:
    eval_output = []
    reasoning_outputs = json.load(open(my_file, 'r'))
    for output in reasoning_outputs:
        eval_output.append(extract_boxed_escaped_string(output))
    
    save_path = transform_path(my_file)
    os.makedirs(save_path.parent, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(eval_output, f, indent=2)