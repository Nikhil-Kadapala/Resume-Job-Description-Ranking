import os
import argparse
import json
import sys
import torch
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Type
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration, AutoTokenizer
from peft import PeftModel, PeftConfig, get_peft_model

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.prompts import SYSTEM_PROMPT, get_test_user_prompt
from utils.parse_resume import parse_resume
from data_structures.analysis_data import ResumeAnalysis

def load_lora_model(base_model_name_or_path, lora_weights_path, device=None):
    """
    Loads a Gemma-3 base model and applies LoRA adapters for inference on a local GPU (if available).
    Args:
        base_model_name_or_path (str): Path or name of the base model (Hugging Face Hub or local).
        lora_weights_path (str): Path to the LoRA adapter weights (directory or file).
        device (str, optional): Device to use ('cuda', 'cpu', etc). Defaults to CUDA if available.
    Returns:
        model: The model with LoRA adapters applied.
        processor: The processor for the model.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Loading Gemma-3 base model from {base_model_name_or_path}")
    # Load Gemma-3 model and processor
    base_model = Gemma3ForConditionalGeneration.from_pretrained(
        base_model_name_or_path,
        torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
        device_map="auto" if device == 'cuda' else None
    ).eval()
    
    processor = AutoProcessor.from_pretrained(base_model_name_or_path)
    
    print(f"Loading LoRA adapters from {lora_weights_path}")
    # Load LoRA adapters
    model = PeftModel.from_pretrained(base_model, lora_weights_path)
    model.eval()
    print(f"Model loaded successfully on {device}")
    return model, processor

def generate_text(model, processor, system_prompt, user_prompt, max_new_tokens=512, temperature=0.3, device=None):
    """
    Runs inference with the Gemma-3 model and LoRA adapters on the provided prompt.
    Args:
        model: The model with LoRA adapters applied.
        processor: The processor for the model.
        system_prompt (str): The system prompt to guide the model.
        user_prompt (str): The user prompt to generate text from.
        max_new_tokens (int): Maximum number of new tokens to generate.
        temperature (float): Temperature for sampling.
        device (str, optional): Device to use. Defaults to CUDA if available.
    Returns:
        str: The generated text.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Format messages for Gemma-3 model
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system_prompt}]
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": user_prompt}]
        }
    ]
    
    # Process input using the processor
    inputs = processor.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=True,
        return_dict=True, 
        return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)
    
    input_len = inputs["input_ids"].shape[-1]
    
    # Generate response
    with torch.inference_mode():
        generation = model.generate(
            **inputs, 
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0),
            top_p=0.95,
        )
        generation = generation[0][input_len:]
    
    return processor.decode(generation, skip_special_tokens=True)

def fetch_resume_data(resume_path: str) -> str:
    """
    Fetches resume data from the given path.

    Args:
        resume_path (str): The path to the resume file.

    Returns:
        str: The text content of the resume.
    """
    resume_content = parse_resume(resume_path)
    return resume_content

def get_lora_resume_analysis(model, processor, sys_prompt: str, user_prompt: str, 
                            response_format: Type[ResumeAnalysis] | ResumeAnalysis) -> str:
    """
    Gets resume analysis using the Gemma-3 LoRA model.
    
    Args:
        model: The LoRA model.
        processor: The processor for the model.
        sys_prompt (str): The system prompt.
        user_prompt (str): The user prompt.
        response_format: The expected response format.
        
    Returns:
        str: The analysis result.
    """
    assert isinstance(sys_prompt, str), "sys_prompt must be a string"
    assert isinstance(user_prompt, str), "user_prompt must be a string"
    assert response_format == ResumeAnalysis or isinstance(response_format, ResumeAnalysis), \
        "response_format must be a ResumeAnalysis class or instance"
    
    try:
        response = generate_text(model, processor, sys_prompt, user_prompt)
        return response
    except Exception as e:
        print(f"Error in get_lora_resume_analysis(): {e}")
        return ""

def rank_resumes_with_lora(base_model_path: str, lora_weights_path: str, 
                         resumes_path: str, jds_path: str, 
                         response_format: ResumeAnalysis) -> Dict[str, List[ResumeAnalysis]]:
    """
    Ranks resumes based on their relevance to the job description using a Gemma-3 LoRA model.

    Args:
        base_model_path (str): Path to the base model.
        lora_weights_path (str): Path to the LoRA weights.
        resumes_path (str): Path to the directory containing resumes.
        jds_path (str): Path to the job descriptions CSV file.
        response_format (ResumeAnalysis): The format class or instance for the response.

    Returns:
        Dict[str, List[ResumeAnalysis]]: A dictionary mapping resume IDs to their analysis results.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the model with LoRA adapters
    model, processor = load_lora_model(base_model_path, lora_weights_path, device)

    resumes_dir = Path(resumes_path)
    if not resumes_dir.is_dir():
        print(f"Error: {resumes_path} is not a valid directory")
        return {}
    
    resume_texts = []
    resume_files = list(resumes_dir.glob("*.docx"))
    
    print(f"Found {len(resume_files)} resume files")
    for file_path in resume_files:
        print(f"Processing resume: {file_path.name}")
        resume_content = fetch_resume_data(file_path)
        resume_texts.append(resume_content)

    jd_df = pd.read_csv(jds_path)
    print(f"Loaded {len(jd_df)} job descriptions")

    analysis_reports: Dict[str, List[ResumeAnalysis]] = {}

    for idx, resume_text in tqdm(enumerate(resume_texts), desc="Analyzing Resumes"):
        res_id = f"Resume_{idx+1}"
        res_data = {
            res_id: [],
        }
    
        for index, jd in tqdm(jd_df.iterrows(), desc=f"Analyzing Job Descriptions for {res_id}"):
            job_description_text = jd['job_description']
            user_prompt = get_test_user_prompt(resume_text, job_description_text)

            try:
                response = get_lora_resume_analysis(model, processor, SYSTEM_PROMPT, user_prompt, response_format)
                res_data[res_id].append(response)
            except Exception as e:
                print(f"Error in getting ranking for {res_id} with job {index}: {e}")

        analysis_reports[res_id] = res_data[res_id]

    return analysis_reports

def main():
    parser = argparse.ArgumentParser(description="Rank resumes using a Gemma-3 LoRA model on a local GPU.")
    parser.add_argument("-r", "--resume", type=str, help="Path to the test data folder with resumes.")
    parser.add_argument("-j", "--jobs", type=str, help="Path to the job description file.")
    parser.add_argument("-b", "--base_model", type=str, help="Path or name of the Gemma-3 base model.")
    parser.add_argument("-l", "--lora", type=str, help="Path to the LoRA adapter weights.")
    args = parser.parse_args()

    # Set default values if not provided
    curr_dir = Path(__file__).parent
    parent_dir = curr_dir.parent.parent
    
    BASE_MODEL = args.base_model if args.base_model else "google/gemma-3-12b-it"
    LORA_WEIGHTS = args.lora if args.lora else parent_dir / "models" / "lora_adapters"
    RESUMES_PATH = args.resume if args.resume else parent_dir / "data" / "test" / "Resumes"
    JDS_PATH = args.jobs if args.jobs else parent_dir / "data" / "test" / "JDs.csv"

    print(f"Base model: {BASE_MODEL}")
    print(f"LoRA weights: {LORA_WEIGHTS}")
    print(f"Resumes path: {RESUMES_PATH}")
    print(f"Job descriptions path: {JDS_PATH}")

    ranked_results = rank_resumes_with_lora(BASE_MODEL, LORA_WEIGHTS, RESUMES_PATH, JDS_PATH, ResumeAnalysis)
    
    # Create the output directory if it doesn't exist
    output_dir = parent_dir / "data" / "results" / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    INFERENCE_FILE_PATH = output_dir / "lora_inference.jsonl"

    with open(INFERENCE_FILE_PATH, 'w', encoding='utf-8') as f:
        for k, v in ranked_results.items():
            json_line = json.dumps({k: v}, ensure_ascii=False)
            f.write(json_line + '\n')
    
    print("Inference completed. Results saved to:", INFERENCE_FILE_PATH)

if __name__ == "__main__":
    main()
