from typing import Dict, List, Any
from pydantic import BaseModel, Field

class Education(BaseModel):
    """
    Pydantic model to represent the education requirements in the job description.
    """
    required_degree: List[str] = Field(description="Required degree for the job. For example: BS, MS, PhD, MBA etc.")
    preferred_degree: List[str] = Field(description="Preferred degree for the job. For example: BS, MS, PhD, MBA etc.")
    required_level: List[str] = Field(description="Required level for the job. For example: Diploma, Bachelor's, Master's, Doctoral, etc.")
    preferred_level: List[str] = Field(description="Preferred level for the job. For example: Diploma, Bachelor's, Master's, Doctoral, etc.")
    required_major: List[str] = Field(description="Required major for the job. For example: Computer Science, Computer Engineering, Business Administration, etc.")
    preferred_major: List[str] = Field(description="Preferred major for the job. For example: Computer Science, Computer Engineering, Business Administration, etc.")

class Experience(BaseModel):
    """
    Pydantic model to represent the experience mentioned in the job description.
    """
    required_years_in_total: int = Field(description="Required total years of experience for the job")
    preferred_years_in_total: int = Field(description="Preferred total years of experience for the job")

class Skills(BaseModel):
    """
    Pydantic model to represent the skills listed in the job description.
    """
    required_technical: List[str] = Field(description="Required technical skills for the job")
    preferred_technical: List[str] = Field(description="Preferred technical skills for the job")
    required_soft: List[str] = Field(description="Required soft skills for the job")
    preferred_soft: List[str] = Field(description="Preferred soft skills for the job")
    required_languages: List[str] = Field(description="Required languages for the job")
    preferred_languages: List[str] = Field(description="Preferred languages for the job")
    required_certifications: List[str] = Field(description="Required certifications for the job")
    preferred_certifications: List[str] = Field(description="Preferred certifications for the job")

class OtherInformation(BaseModel):
    """
    Pydantic model to represent other information mentioned in the job description.
    """
    salary: str = Field(description="Salary range for the job")
    benefits: List[str] = Field(description="Benefits mentioned in the job description")
    bonus_qualifications: List[str] = Field(description="Bonus qualifications mentioned in the job description that are not mandatory but potentially beneficial")
    relocation_assistance: bool = Field(description="Whether relocation assistance is provided for the job")

class JobDescription(BaseModel):
    """
    Pydantic model to represent the job description.
    """
    job_title: str = Field(description="Title/Position/Role of the job. For example: Software Engineer, Data Scientist, etc.")
    location: List[str] = Field(description="Location of the job")
    job_type: str = Field(description="Type of job. For example: Full-time, Part-time, Contract, etc.")
    work_type: str = Field(description="Type of work. For example: Remote, On-site, Hybrid, etc.")
    EDUCATION: Education
    EXPERIENCE: Experience
    SKILLS: Skills
    OTHER_INFORMATION: OtherInformation