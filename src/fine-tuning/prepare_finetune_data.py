import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd
from typing import List
# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.prompts import SYSTEM_PROMPT, get_distill_user_prompt

def extract_distillation_results(distill_data_path: str) -> List[str]:
    distill_results = []
    with open(distill_data_path, 'r', encoding='utf-8') as distill:
        for line_num, line in enumerate(distill, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                distill_results.append(obj)
            except JSONDecodeError as e:
                print(f"Skipping malformed JSON on line {line_num}: {e}")
    return distill_results

def create_distillation_user_prompt(train_file_path: str) -> List[str]:
    """Construct the user prompt for each entry in the training data."""

    data = pd.read_csv(train_file_path)
    user_prompts_list = []

    for index, row in data.iterrows():

        user_prompt = get_distill_user_prompt(row['resume_text'], row['job_description_text'], row['label'])
        user_prompts_list.append(user_prompt)

    return user_prompts_list

def create_instruction_dataset(system_prompt, user_prompts_list, responses, FINE_TUNE_FILE_PATH):
    """Create the instruction fine-tuning dataset."""
    try:
        
        with open(FINE_TUNE_FILE_PATH, 'w', encoding='utf-8') as finetune:
            for index, (user_prompt, response) in enumerate(zip(user_prompts_list, responses)):
                try:
                    json_data = {
                        "prompt": f"System: {system_prompt}User: {user_prompt}",
                        "completion": str(response)
                    }
                    #json_data["completion"] = json.dumps(json_data["completion"], separators=(',', ':'), ensure_ascii=False)
                    json_line = json.dumps(json_data, separators=(',', ':'), ensure_ascii=False)
                    json.loads(json_line)
                    finetune.write(json_line + '\n')
                except Exception as e:
                    print(f"Error processing line {index}: {str(e)}")
                    continue
        
        print(f"Created instruction fine-tuning data with {len(user_prompts_list)} entries.")
        print(f"Fine-tuning data saved to {FINE_TUNE_FILE_PATH}")
    
    except Exception as e:
        print(f"Error creating instruction fine-tuning data: {e}")

def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data for the model.")
    parser.add_argument("-i", "--distill", type=str, help="Path to the distillation results data file (JSONL).")
    parser.add_argument("-o", "--finetune", type=str, help="Path to the output or fine-tuning data file (JSONL).")
    parser.add_argument("-t", "--train", type=str, help="Path to the training data file (CSV).")
    args = parser.parse_args()

    DISTILL_DATA_PATH: str
    FINE_TUNE_FILE_PATH: str
    
    current_dir = Path(__file__).parent
    project_root = current_dir.parent

    DISTILL_DATA_PATH = args.distill if args.distill else project_root.parent / "data/results/distillation" / "distillation_results.jsonl"
    TRAIN_FILE_PATH = args.train if args.train else project_root.parent / "data" / "train.csv"
    FINE_TUNE_FILE_PATH = args.finetune if args.finetune else project_root.parent / "data/fine-tuning" / "fine_tune_data.jsonl"

    responses_list = extract_distillation_results(DISTILL_DATA_PATH)
    user_prompts_list = create_distillation_user_prompt(TRAIN_FILE_PATH)

    create_instruction_dataset(SYSTEM_PROMPT, user_prompts_list, responses_list, FINE_TUNE_FILE_PATH)

if __name__ == "__main__":
    main()