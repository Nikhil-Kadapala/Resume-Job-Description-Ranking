from typing import Dict, List, Any
from pydantic import BaseModel, Field

class Education(BaseModel):
    """
    Pydantic model to represent the education of the resume.
    """
    degree: List[str] = Field(description="Degree of the education. For example: BS, MS, PhD, MBA, etc.")
    level: List[str] = Field(description="Level of the education. For example: Diploma, Bachelor's, Master's, Doctoral, etc.")
    major: List[str] = Field(description="Major of the education. For example: Computer Science, Business Administration, etc.")

class Experience(BaseModel):
    """
    Pydantic model to represent the experience of the resume.
    """
    years_in_total: int = Field(description="Total years of experience")
    years_in_current_company: int = Field(description="Years of experience in the current company")
    current_employer: List[str] = Field(description="Name of the company where the candidate is currently employed or the most recent employer")
    position: List[str] = Field(description="Position or role at the current company or the most recent position")
    duration: List[str] = Field(description="Duration at the current company or the most recent position")

class Skills(BaseModel):
    """
    Pydantic model to represent the skills of the resume.
    """
    technical: List[str] = Field(description="Technical skills")
    soft: List[str] = Field(description="Soft skills")
    languages: List[str] = Field(description="Languages spoken")
    certifications: List[str] = Field(description="Certifications obtained")

class OtherInformation(BaseModel):
    """
    Pydantic model to represent other information of the resume.
    """
    awards_and_achievements: List[str] = Field(description="Awards and achievements of the candidate. For example: Employee of the Month, Best Project Award, Dean's List, Fellowships, etc.")
    publications: List[str] = Field(description="Publications by the candidate")
    projects: List[str] = Field(description="Projects worked on by the candidate")
    volunteering: List[str] = Field(description="Volunteering experience of the candidate")
    leadership: List[str] = Field(description="Leadership experience of the candidate")

class ContactInformation(BaseModel):
    """
    Pydantic model to represent the contact information of the resume.
    """
    email: List[str] = Field(description="Email address of the candidate")
    phone: List[str] = Field(description="Phone number of the candidate")
    address: List[str] = Field(description="Address of the candidate")
    website: List[str] = Field(description="Website of the candidate or URL to the LinkedIn/GitHub profile")

class Qualifications(BaseModel):
    """
    Pydantic model to represent the qualifications of the resume.
    """
    SKILLS: Skills
    EDUCATION: Education
    EXPERIENCE: Experience
    OTHER_INFORMATION: OtherInformation
    CONTACT_INFORMATION: ContactInformation

class Resume(BaseModel):
    """
    Pydantic model to represent the resume.
    """
    summary: str = Field(description="Summary of the resume")
    job_title: str = Field(description="Current Job title of the candidate or the position they are applying for")
    qualifications: Qualifications = Field(description="Qualifications of the resume")