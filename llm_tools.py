import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import re
import json
from typing import Type, List
class LLM:
    def __init__(self, model_path, device: str, cache_dir = None):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=cache_dir)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, cache_dir=cache_dir).to(device).eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model.generation_config is not None and self.model.generation_config.pad_token_id is None:
            self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id


    def call_llm_with_text(self, text: str, max_length) -> str:
        messages = self.call_llm([{"role": "user", "content": text}], max_length)
        return messages[-1]["content"]

    def call_llm(self, messages:List, max_length: int):
        messages = messages.copy()
        text  = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        model_inputs = self.tokenizer(text , return_tensors="pt", padding=True).to(self.model.device)
        with torch.inference_mode():
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=max_length)
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        messages.append({"role": "assistant", "content": content})
        return messages

    def structured_output(self, query, max_length, structure: Type[BaseModel], retry_count=0):
        system = (f"Answer the user query. Output your answer as JSON "
                  f"that matches the given schema: ```json\n{structure.model_json_schema()}\n```. "
                  f"Make sure to wrap the answer in ```json and ``` tags")
        messages = [{"role": "system", "content": system}, {"role": "user", "content": query}]

        for i in range(retry_count+1):
            messages = self.call_llm(messages, max_length)
            response = messages[-1]["content"]
            try:
                matches = re.findall(r"```json(.*?)```", response, re.DOTALL)
                obj = json.loads(matches[-1])
                obj = structure.model_validate(obj).model_dump()
                return {
                    "message": messages,
                    "retry count": i,
                    "structured output": obj
                }
            except Exception:
                pass

            new_query = (f"The output is invalid, please try again. "
                             f"You should answer the question in JSON that matches the given schema: "
                             f"```json\n{structure.model_json_schema()}\n```. "
                             f"Make sure to wrap the answer in ```json and ``` tags")
            messages.append({"role": "user", "content": new_query})

        return {
                    "message": messages,
                    "retry count": retry_count,
                    "structured output": None
                }

