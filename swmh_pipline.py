from pyexpat.errors import messages

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
    label: Literal["no mental disorder", "suicide", "depression", "anxiety", "bipolar"] = Field(...,description="Single most likely primary label extracted from the content.")
    explanation: str = Field(..., description="A plain-text summary of explanation about the decision making extracted from the content.")
    confidence: Literal["low", "medium", "high"] = Field(...,description="The confidence level extracted from the content.")



def inference(llm, post, log = False):
    if log:
        print(f"Post: {post}\n\n")
    single_symptom_result = {}
    for key, prompt_file in single_symptom_prompt.items():
        prompt = imhi_dataset.apply_prompts({"post": post}, prompt_file)
        output = llm.call_llm_with_text(prompt, 2048)
        output = output.split("\n")[-1]
        single_symptom_result[key] = output
        if log:
            print(f"[{key} result]: \n{output}\n\n")
    prompt_for_summary = imhi_dataset.apply_prompts(single_symptom_result|{"post": post}, "prompts/swmh/final_summary.txt")
    output = llm.call_llm_with_text(prompt_for_summary , 2048)
    if log:
        print(f"Output: \n{output}\n\n")
    prompt = imhi_dataset.apply_prompts({"analysis": output, "post": post}, "prompts/swmh/structure_output.txt")
    structured_output = llm.structured_output(prompt, 2048,  MentalDisorderReport, retry_count= 3)
    if log:
        if structured_output["structured_output"] is None:
            print(f"[error to get structured output]{structured_output["message"][-1]["content"]}\n\n")
        else:
            print(f"\nStructured Output[{structured_output["retry_count"]}]:\n{structured_output["structured_output"]}\n\n")

    return prompt_for_summary ,output, structured_output["structured_output"]







