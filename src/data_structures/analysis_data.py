from typing import Dict, List, Any
from enum import Enum
from pydantic import BaseModel, Field
from .resume_data import Resume
from .jd_data import JobDescription

class ClassEnum(str, Enum):
    """
    Enum to represent the classification of the resume.
    """
    GOOD_FIT = "Good Fit"
    NOT_FIT = "Not Fit"
    PARTIAL_FIT = "Partial Fit"

class ResumeAnalysis(BaseModel):
    """
    Pydantic model to represent the analysis of the resume.
    """
    summary: str = Field(description="Summary of the resume that only includes the most relevant information. For example: Name, Current Position, Current Company, Current Location, Years of Experience, top 5 skills.")
    classification: ClassEnum = Field(description="Classification of the resume. For example: Good Fit, Not Fit, Partial Fit")
    overall_score: float = Field(description="Overall score of the resume on a scale of 1 to 100 based on how well it matches the job description. For example: 85")
    rationale: str = Field(description="A clear, second-person explanation directly addressing the owner of the resume why they were classified as Good Fit, Partial Fit, or Not Fit. For example: \'You are a strong match because you bring 7 years of backend engineering experience...\' or \'You are currently not a strong match because your background focuses on UX design, while this role requires backend expertise.\' Focus on strengths, gaps, and the reasoning behind the score.")
    suggestions: str = Field(description="Personalized advice written directly to the candidate, using second-person tone. For example: \"You should improve your resume by adding certifications in cloud technologies.\" or \"Consider adding certifications in cloud technologies to strengthen your resume.\" Instead of: \"The candidate should add more experience in the required skills, obtain certifications, or consider taking courses related to a specific area.\"")
    matching_skills: List[str] = Field(description="List of skills that match the job description. For example: ['Python', 'SQL', 'Machine Learning', 'Data Visualization', 'AWS']")
    missing_skills: List[str] = Field(description="List of skills that are missing in the resume but are required in the job description. For example: ['Java', 'C++', 'Project Management']")
    resume: Resume = Field(description="Structured data extracted from the Resume of the candidate")
    job_description: JobDescription = Field(description="Structured data extracted from the Job Description")