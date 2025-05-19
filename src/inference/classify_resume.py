import os
import argparse
import json
import sys
import pandas as pd
from pathlib import Path
from together import Together
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.prompts import SYSTEM_PROMPT, get_test_user_prompt
from utils.parse_resume import parse_resume
from utils.get_resume_analysis import get_resume_analysis
from utils.get_model_response import get_model_response
from data_structures.analysis_data import ResumeAnalysis

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

def load_annotations(annotations_file_path: str) -> Dict[str, Any]:
    """
    Load annotations from a JSON file.

    Args:
        annotations_file_path (str): Path to the annotations JSON file.

    Returns:
        Dict[str, Any]: Dictionary containing the loaded annotations.
    """
    if not annotations_file_path.exists():
        print(f"Error: {annotations_file_path} does not exist.")
        return {}
            
    with open(annotations_file_path, 'r') as f:
        annotations = json.load(f)
        return annotations

def classify_resume(model: str, data_path: str, response_format: ResumeAnalysis) -> Dict[str, List[ResumeAnalysis]]:
    """
    Ranks resumes based on their relevance to the job description.

    Args:
        model (str): The model to be used for ranking.
        train_data (pd.DataFrame): The training data containing resumes and job descriptions.

    Returns:
        List[ResumeAnalysis]: A list of ResumeAnalysis objects containing the ranking results.
    """

    resumes_dir = Path(resumes_path)

    # Check if the directory exists
    if not resumes_dir.is_dir():
        print(f"Error: {resumes_path} is not a valid directory")
        return
    
    resume_texts = []

    for file_path in resumes_dir.glob("*.docx"):
        resume_content = fetch_resume_data(file_path)
        resume_texts.append(resume_content)

    jd_df = pd.read_csv(jds_path)

    analysis_reports: Dict[List[ResumeAnalysis]] = {}

    for idx, resume_text in enumerate(resume_texts[0:2]):
        
        res_id = f"Resume_{idx}"
        res_data = f"{res_id}_data" 
        res_data = {
            res_id: [],
        }
    
        for index, jd in jd_df.iterrows():

            job_description_text = jd['job_description']
            user_prompt = get_test_user_prompt(resume_text, job_description_text)

            try:
                response = get_resume_analysis(model, SYSTEM_PROMPT, user_prompt, response_format)
                res_data[res_id].append(response)
            except Exception as e:
                print(f"Error in getting ranking module: {e}")
        analysis_reports[res_id] = res_data
    return analysis_reports

def main():
    parser = argparse.ArgumentParser(description="Rank resumes based on their relevance to the job description.")
    parser.add_argument("-r", "--resume", type=str, help="Path to the test data folder with resumes.")
    parser.add_argument("-j", "--jobs", type=str, help="Path to the job description file.")
    parser.add_argument("-m", "--model", type=str, help="Model to be used for ranking.")
    args = parser.parse_args()

    RESUMES_PATH: str
    JDS_PATH: str
    MODEL: str
    MODEL = args.model if args.model else "Nikhil_Kadapala/gemma-3-12b-it-Distill-Gemini-2.5-Flash-preview-04-17-8e64d62f-8b157c3f"

    curr_dir = Path(__file__).parent
    parent_dir = curr_dir.parent.parent
    DATA_PATH = args.resume if args.resume else parent_dir / "data" / "test" / "test.csv"

    # Rank the resumes
    ranked_results = classify_resume(MODEL, DATA_PATH, ResumeAnalysis)
    #print(ranked_results.items())
    
    INFERENCE_FILE_PATH = parent_dir / "data" / "results" / "inference" / "inference.jsonl"

    with open(INFERENCE_FILE_PATH, 'w', encoding='utf-8') as f:
        for k ,v in ranked_results.items():
            json_line = json.dumps(v, ensure_ascii=False)
            f.write(json_line + '\n')
    
    ANNOTATIONS_FILE_PATH = parent_dir / "data" / "test" / "annotations.json"
    annotations = load_annotations(ANNOTATIONS_FILE_PATH)
    
if __name__ == "__main__":
    main()