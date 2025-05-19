import os
from together import Together
from typing import Any
import argparse
import subprocess
import time

try:
    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
except Exception as e:
    print(f"Error: {e}")
    exit()

try:
    WANDB_API_KEY = os.getenv("WANDB_API_KEY")
except Exception as e:
    print(f"Error: {e}")
    exit()

def create_finetune_job(model_name: str, train_file_id: str, val_file_id: str, suffix: Any):
    suffix: str
    client = Together(api_key = TOGETHER_API_KEY)
    try:

        response = client.fine_tuning.create(
            training_file = train_file_id,
            validation_file = val_file_id,
            train_on_inputs=True,
            model = model_name,
            lora = True,
            lora_r = 16,
            lora_alpha = 32,
            lora_dropout = 0.1,
            n_epochs = 3,
            n_checkpoints = 2,
            batch_size = 8,
            learning_rate = 2e-5,
            suffix = suffix,
            wandb_api_key = WANDB_API_KEY,
            wandb_project_name = "Resume-JD-Ranking",
            wandb_name = "lora-finetune-gemma-3-12b-it",
        )

        return response
    
    except Exception as e:
        print("Error creating fine-tune job:", e)
        exit()
    
def main():
    parser = argparse.ArgumentParser(description="Evaluating the fine-tuned model's performance with METEOR Score on the dev set.")
    parser.add_argument("-tf", "--trfileID", type=str, help="file-id of the training file uploaded to together.ai")
    parser.add_argument("-vf", "--valfileID", type=str, help="file-id of the validation file uploaded to together.ai")
    parser.add_argument("-m", "--model", type=str, help="Fine-tuned Model name.")
    parser.add_argument("-s", "--suffix", type=str, help="Suffix for the model name.")
    args = parser.parse_args()
    
    finetune_response: Any
    TRAIN_FILE_ID: str
    VAL_FILE_ID: str
    MODEL: str
    SUFFIX: str
    
    if args.trfileID is None:
        print("Please provide the fiLe-id. It looks something like this: file-5e32a8e6-72b3-485d-ab76-71a73d9e1f5b")
        exit()
    else:
        TRAIN_FILE_ID = args.trfileID
    
    if args.valfileID is None:
        print("Please provide the fiLe-id. It looks something like this: file-5e32a8e6-72b3-485d-ab76-71a73d9e1f5b")
        exit()
    else:
        VAL_FILE_ID = args.valfileID

    if args.model is None:
        print(f"Please provide the model name to fine-tune.\n Go to https://docs.together.ai/docs/fine-tuning-models to see the list of models available for fine-tuning.")
        exit()
  
    SUFFIX = "my-fine-tuned-model" if args.suffix is None else args.suffix
    MODEL = args.model
        
    finetune_response = create_finetune_job(MODEL, TRAIN_FILE_ID, VAL_FILE_ID, SUFFIX)

    print(f"Fine-tuning job ID:\n{finetune_response}")

    ft_id = finetune_response.id

    while True:
        process = subprocess.Popen(['together', 'fine-tuning', 'list-events', ft_id], stdout=subprocess.PIPE, text=True)
        output, _ = process.communicate()
        print(output)
        if "Job finished" in output:
            break
        time.sleep(60)

    print(f"Fine-tuning job completed successfully.\n visit https://api.together.ai/fine-tuning to download the checkpoints or deploy a dedicated endpoint for inference.")

if __name__ == "__main__":
    main()
