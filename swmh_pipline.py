import imhi_dataset
from pydantic import BaseModel, Field
from typing import Literal

single_symptom_prompt = {
    'suicide':"prompts/swmh/suicide_check.txt",
    'depression':"prompts/swmh/depression_check.txt",
    'anxiety':"prompts/swmh/anxiety_check.txt",
    'bipolar':"prompts/swmh/bipolar_check.txt",
}

class MentalDisorderReport(BaseModel):
    disorder: Literal["no mental disorder", "suicide", "depression", "anxiety", "bipolar"] = Field(...,description="Single most likely primary label extracted from the content.")
    explanation: str = Field(..., description="A plain-text summary of explanation about the decision making extracted from the content.")
    confidence: Literal["Low", "Medium", "High"] = Field(...,description="The confidence level extracted from the content.")

def inference(llm, post):
    single_symptom_result = {}
    for key, prompt_file in single_symptom_prompt.items():
        prompt = imhi_dataset.apply_prompts({"post": post}, prompt_file)
        output = llm.call_llm_with_text(prompt, 2048)
        output = output.split("\n")[-1]
        single_symptom_result[key] = output

    prompt = imhi_dataset.apply_prompts(single_symptom_result|{"post": post}, "prompts/swmh/final_summary.txt")
    print(f"Query: \n{prompt}\n\n")

    output = llm.call_llm_with_text(prompt, 2048)
    print(f"Output: \n{output}")

    prompt = imhi_dataset.apply_prompts({"content": output}, "prompts/swmh/structure_output.txt")
    structured_output = llm.structured_output(prompt, 2048,  MentalDisorderReport, 2)["structured output"]

    print(f"\nStructured Output:\n{structured_output}")
    return structured_output







