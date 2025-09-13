import swmh_pipline
import llm_tools
import imhi_dataset

llm = llm_tools.LLM(model_path="meta-llama/Llama-3.2-1B-Instruct", device="cuda")
dataset = imhi_dataset.get_dataset("dataset/test/swmh.csv")
swmh_pipline.inference(llm, dataset[3]["post"])
