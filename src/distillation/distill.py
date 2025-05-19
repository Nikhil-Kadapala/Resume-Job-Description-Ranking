import argparse
import json
import pandas as pd
import sys
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils.prompts import SYSTEM_PROMPT, get_distill_user_prompt
from utils.parse_resume import parse_resume
from data_structures.analysis_data import ResumeAnalysis, ClassEnum
from data_structures.resume_data import Resume
from data_structures.jd_data import JobDescription
from utils.get_teacher_response import get_teacher_response

def create_fallback_analysis(resume_text: str, job_description_text: str, classification_label: str) -> ResumeAnalysis:
    """
    Creates a fallback ResumeAnalysis object when model response fails.
    
    Args:
        resume_text (str): The text of the resume.
        job_description_text (str): The text of the job description.
        classification_label (str): The classification label for the resume.
        
    Returns:
        ResumeAnalysis: A basic ResumeAnalysis object with fallback values.
    """
    class_enum = ClassEnum.GOOD_FIT
    if classification_label.lower() == "not fit":
        class_enum = ClassEnum.NOT_FIT
    elif classification_label.lower() == "partial fit":
        class_enum = ClassEnum.PARTIAL_FIT
    
    return ResumeAnalysis(
        summary=f"Failed to analyze resume - fallback summary created",
        classification=class_enum,
        overall_score=50.0,
        rationale="Model failed to generate analysis. This is a fallback response.",
        suggestions="Please try again with a different model or check the resume and job description.",
        matching_skills=["Failed to extract"],
        missing_skills=["Failed to extract"],
        resume=Resume(
            name="Failed to extract",
            contact_information={},
            education=[],
            work_experience=[],
            skills=[],
            certifications=[],
            full_text=resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
        ),
        job_description=JobDescription(
            title="Failed to extract",
            company_name="Failed to extract",
            location="Failed to extract",
            required_skills=[],
            preferred_skills=[],
            responsibilities=[],
            qualifications=[],
            full_text=job_description_text[:500] + "..." if len(job_description_text) > 500 else job_description_text
        )
    )
    
def analyze_resume(model: str, resume_text: str, job_description_text: str, classification_label: str, response_format: ResumeAnalysis) -> ResumeAnalysis:
    """
    Analyzes a resume against a job description and returns a ResumeAnalysis object.

    Args:
        model (str): The model to be used for analysis.
        resume_text (str): The text of the resume.
        job_description_text (str): The text of the job description.
        classification_label (str): The classification label for the resume.

    Returns:
        ResumeAnalysis: An object containing the analysis results.
    """
    user_prompt = get_distill_user_prompt(resume_text, job_description_text, classification_label)

    try:
        response = get_teacher_response(model, SYSTEM_PROMPT, user_prompt, response_format)
        
        # Check if response is None or an empty string
        if response is None or response == "":
            print(f"Warning: Empty response received for a resume. Creating fallback ResumeAnalysis object.")
            return create_fallback_analysis(resume_text, job_description_text, classification_label)
            
        return response

    except Exception as e:
        print(f"Error in analyze_resume: {e}")
        return create_fallback_analysis(resume_text, job_description_text, classification_label)

def start_distillation(model:str, train_data: pd.DataFrame, response_format: ResumeAnalysis, results_file_path: Path, classes_file_path: Path) -> None:
    """
    Starts the distillation process for the given model and training data.
    Saves results incrementally as they are generated.

    Args:
        model (str): The model to be used for distillation.
        train_data (pd.DataFrame): The training data containing resumes and job descriptions.
        response_format: The response format class.
        results_file_path (Path): Path where to save the results.
        classes_file_path (Path): Path where to save the classification results.
    """
    # Create results directory if it doesn't exist
    results_dir = results_file_path.parent
    if not results_dir.exists():
        results_dir.mkdir(parents=True, exist_ok=True)
    
    # Open files for appending results incrementally
    with open(results_file_path, 'w') as results_file:
        predicted_labels = []
        
        for index, row in tqdm(train_data.iterrows(), total=train_data.shape[0], desc="Distilling Resumes"):
            try:
                resume_text = row['resume_text']
                job_description_text = row['job_description_text']
                classification_label = row['label']
                
                # Analyze resume
                analysis = analyze_resume(model, resume_text, job_description_text, classification_label, response_format)
                
                # Save result immediately
                if analysis is not None:
                    json_line = json.dumps(analysis.model_dump())
                    results_file.write(json_line + '\n')
                    results_file.flush()  # Force write to disk
                    
                    # Record classification
                    predicted_labels.append(analysis.classification.value)
                    
                    # Save classifications periodically (every 10 items)
                    if len(predicted_labels) % 10 == 0:
                        classifications = {"predicted_labels": predicted_labels}
                        pd.DataFrame(classifications).to_csv(classes_file_path, index=False)
                        print(f"Progress: {len(predicted_labels)}/{train_data.shape[0]} items processed")
            
            except Exception as e:
                print(f"Error processing item {index}: {e}")
                # Continue with next item
        
        # Final save of classifications
        classifications = {"predicted_labels": predicted_labels}
        pd.DataFrame(classifications).to_csv(classes_file_path, index=False)
        
    print(f"Distillation completed. Processed {len(predicted_labels)} out of {train_data.shape[0]} items.")
    print(f"Results saved to {results_file_path} and classifications saved to {classes_file_path}")

def main():
    parser = argparse.ArgumentParser(
        description="""Distilling knowledge from a resume and job description against to support classification.
        Accepted model names: gemini-2.5-flash-preview-04-17, gemini-2.0-flash, gemini-2.0-flash-lite.
        Accepted data file format: CSV file only.
        Note: If no arguments are provided, default values will be used."""
    )
    
    parser.add_argument("-m", "--model", nargs='*', type=str, help="Model name.")
    parser.add_argument("-tr", "--train", nargs='*', type=str, help="Prompt style.")
    
    args = parser.parse_args()

    MODEL: str 
    FILE_PATH: str
    TRAIN_DATA: pd.DataFrame
      
    if args.model:
        MODEL = args.model
    else:
        MODEL = "gemini-2.5-flash-preview-04-17"
        print(f"Using default model: {MODEL}")
    
    current_dir = Path(__file__).resolve().parent

    TRAIN_FILE_PATH = Path(args.train[0]) if args.train else current_dir.parent.parent / "data" / "train.csv"

    if not TRAIN_FILE_PATH.exists():
        print(f"Error: Data file not found at '{TRAIN_FILE_PATH}'")
        print("Please specify the correct path using the -tr or --train parameter")
        return

    TRAIN_DATA = pd.read_csv(TRAIN_FILE_PATH)

    if TRAIN_DATA.empty:
        print(f"Error: Data file '{TRAIN_FILE_PATH}' is empty.")
        return

    RESULTS_FILE_PATH = current_dir.parent.parent / "data/results/distillation" / "distillation_results.jsonl"
    CLASSES_FILE_PATH = current_dir.parent.parent / "data/results/distillation" / "distillation_classes.csv"

    start_distillation(
        model=MODEL, 
        train_data=TRAIN_DATA, 
        response_format=ResumeAnalysis,
        results_file_path=RESULTS_FILE_PATH,
        classes_file_path=CLASSES_FILE_PATH
    )

if __name__ == "__main__":
    main()