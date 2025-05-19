import asyncio
import torch
from transformers import AutoModelForSequenceClassification
"""
def check_faithfulness(retrieved_contexts: str, response: str):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained('vectara/hallucination_evaluation_model', trust_remote_code=True)

    model.to(device)
    
    pairs = [ (retrieved_contexts, response) ]
    
    score = model.predict(pairs)

    return score
"""
from ragas.dataset_schema import SingleTurnSample
from ragas.metrics import FaithfulnesswithHHEM
from ragas.llms import BaseRagasLLM

async def check_faithfulness(retrieved_contexts: str, response: str):
    sample = SingleTurnSample(
        user_input="Evaluate the candidate's suitability for the Basel Business Analyst role.",
        response="You are classified as Not Fit for this Basel Business Analyst role because your experience...",
        retrieved_contexts=[
            "Experienced professional with 16 years in accounting, business management, and administration. Most recent role as Accountant at Aspirus. Skills include Accounting, Excel, Project Management, Staff Management."
        ]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm = BaseRagasLLM().generate(prompt=sample.user_input, retrieved_contexts=sample.retrieved_contexts)
    scorer = FaithfulnesswithHHEM(llm=llm, device=device)
    score = await scorer.single_turn_ascore(sample)
    print(f"Faithfulness Score: {score}")
    return score
