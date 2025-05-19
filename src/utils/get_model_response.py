import os
import json
from typing import Any, Dict, List, Type
from together import Together
from google import genai
from google.genai import types
from data_structures.analysis_data import ResumeAnalysis

    
def get_model_response(model:str, sys_prompt:str, user_prompt:str, response_format:Type[ResumeAnalysis] | ResumeAnalysis)->str:
    """ This function takes a user prompt and a system prompt, and returns the generated response of the model specified.
    
    Args:
        model (str): The model to be used for analysis. Options:"Any model available through Google's Gemini API or Together.ai API".
        user_prompt (str): The user's input prompt with the task query.
        sys_prompt (str): The system prompt to guide the model's response.
        response_format (Type[ResumeAnalysis] | ResumeAnalysis): The format class or instance of response expected from the model.

    Returns:
        str: The generated response from the model.
    
    """
    assert isinstance(model, str), "model name must be a string"
    assert isinstance(user_prompt, str), "user_prompt must be a string"
    assert isinstance(sys_prompt, str), "sys_prompt must be a string"
    assert response_format == ResumeAnalysis or isinstance(response_format, ResumeAnalysis), "response_format must be a ResumeAnalysis class or instance"
    
    analysis: ResumeAnalysis = None
    
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    except Exception as e:
        print(f"Error: {e}")
        exit()

    client = genai.Client(api_key=GEMINI_API_KEY)
       
    try:

        response = client.models.generate_content(
            model=model,
            contents=[types.Part.from_text(text=user_prompt)],
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                response_mime_type='application/json',
                response_schema=response_format,
                temperature=0.1
            ),
        )
        
        analysis = response.parsed

    except Exception as e:
        
        print("Error in get_model_response():", e)     
        analysis = ""
            
    return analysis

