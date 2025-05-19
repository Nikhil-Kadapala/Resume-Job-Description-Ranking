import os
import argparse
import json
import sys
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.prompts import SYSTEM_PROMPT, get_test_user_prompt
from utils.parse_resume import parse_resume
from utils.get_resume_analysis import get_resume_analysis
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


def rank_resumes(model: str, resumes_path: str, jds_path: str,  response_format: ResumeAnalysis, output_file_path: Path) -> Dict[str, List[ResumeAnalysis]]:
    """
    Ranks resumes based on their relevance to the job description.

    Args:
        model (str): The model to be used for ranking.
        train_data (pd.DataFrame): The training data containing resumes and job descriptions.

    Returns:
        List[ResumeAnalysis]: A list of ResumeAnalysis objects containing the ranking results.
    """

    resumes_dir = Path(resumes_path)

    if not resumes_dir.is_dir():
        print(f"Error: {resumes_path} is not a valid directory")
        return
    
    # Store both filename and content as tuples
    resume_data = []

    import re

    def extract_number(filename):
        match = re.match(r"(\d+)", filename)
        return int(match.group(1)) if match else float('inf')

    docx_files = sorted(resumes_dir.glob("*.docx"), key=lambda x: extract_number(x.stem))
    for file_path in docx_files:
        resume_content = fetch_resume_data(file_path)
        resume_data.append((file_path.name, resume_content))

    jd_df = pd.read_csv(jds_path)

    analysis_reports: Dict[str, List[str]] = {}

    with open(output_file_path, 'w', encoding='utf-8') as f:

        for idx, (filename, resume_text) in tqdm(enumerate(resume_data[5:6]), desc="Analyzing Resumes and generating analysis reports"):
            
            # Use original filename in the ID
            res_id = f"Resume_{idx+1}"
            # Include filename in the data structure
            res_data = f"{res_id}_data" 
            res_data = {
                res_id: [],
            }
        
            for index, jd in tqdm(jd_df.iterrows(), desc=f"Analyzing Job Descriptions for {res_id}"):

                job_description_text = jd['job_description']
                user_prompt = get_test_user_prompt(resume_text, job_description_text)
                if res_id not in analysis_reports:
                    analysis_reports[res_id] = []
                analysis_reports[res_id].append(user_prompt)
                response = get_resume_analysis(model, SYSTEM_PROMPT, user_prompt, response_format)
                res_data[res_id].append(response)

            json_line = json.dumps(res_data, ensure_ascii=False)
            f.write(json_line + '\n')
            f.flush()
            
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
    RESUMES_PATH = args.resume if args.resume else parent_dir / "data" / "test" / "Resumes"
    JDS_PATH = args.jobs if args.jobs else parent_dir / "data" / "test" / "JDs.csv"
    INFERENCE_FILE_PATH = parent_dir / "data" / "results" / "inference" / "inference.jsonl"

    ranked_results = rank_resumes(MODEL, RESUMES_PATH, JDS_PATH, ResumeAnalysis, INFERENCE_FILE_PATH)
    
    print("Inference completed. Results saved to: ", INFERENCE_FILE_PATH)
    
if __name__ == "__main__":
    main()