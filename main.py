import swmh_pipline
import llm_tools
import imhi_dataset
import pandas as pd
import os

model_path = input("Please enter the path of your model: ")
llm = llm_tools.LLM(model_path=model_path, device="cuda", cache_dir ="my_model_cache")

dataset = imhi_dataset.get_dataset("dataset/test/swmh.csv")

result = {"query":[], "label": [], "output": [], "structured_output":[]}

for index, data in enumerate(dataset):
    query, output, structured_output = swmh_pipline.inference(llm, data["post"],  True)
    result["query"].append(query)
    result["label"].append(data["label"])
    result["output"].append(output)
    result["structured_output"].append(structured_output)
    if (index + 1) % 100 == 0:
        df = pd.DataFrame(result, index=None)
        os.makedirs("model_result", exist_ok=True)
        df.to_csv(f"model_result/{model_path.replace("/", "--")}.csv", index=False)


df = pd.DataFrame(result, index=None)
os.makedirs("model_result", exist_ok=True)
df.to_csv(f"model_result/{model_path.replace("/", "--")}.csv", index=False)


