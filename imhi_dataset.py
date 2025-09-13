import re
from datasets import Dataset


def apply_prompts(data, prompt_file):
    with open(prompt_file, "r", encoding="utf-8") as f:
        template_prompt = f.read()
    def replace_placeholder(match):
        key = match.group(1).strip()
        keys = key.split("/")
        output = data
        for key in keys:
            output = output[key]
        return str(output)
    return re.sub(r"\[([^\]]+)\]", replace_placeholder, template_prompt)


def get_dataset(dataset_file):
    return Dataset.from_csv(dataset_file)


def get_standard_labels(dataset_name: str):
    if dataset_name == 'swmh':
        standard_labels = ['no mental disorder', 'suicide', 'depression', 'anxiety', 'bipolar']
    elif dataset_name == 't-sid':
        standard_labels = ['no mental disorder', 'depression', 'suicide or self-harm', 'ptsd']
    elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
        standard_labels = ["false", "true"]
    elif dataset_name == 'SAD':
        standard_labels = ['other causes', 'school', 'financial problem', 'family issues', 'social relationships', 'work', 'health issues', 'emotional turmoil', 'everyday decision making']
    elif dataset_name == "CAMS":
        standard_labels = ['none', 'bias or abuse', 'jobs and career', 'medication', 'relationship', 'alienation']
    else:
        raise NameError(f"{dataset_name} is not a valid dataset name")
    return standard_labels


def get_search_labels(dataset_name: str):
    if dataset_name == 'swmh':
        search_labels = ['no mental', 'suicide', 'depression', 'anxiety', 'bipolar']
    elif dataset_name == 't-sid':
        search_labels = ['no mental', 'depression', ['suicide', 'self-harm'], 'ptsd']
    elif dataset_name in ['CLP', 'DR', 'dreaddit', 'loneliness', 'Irf', 'MultiWD']:
        search_labels = ["false", "true"]
    elif dataset_name == 'SAD':
        search_labels = ['other', 'school', 'financial', 'family issue', 'social', 'work', 'health', 'emotional', 'decision']
    elif dataset_name == "CAMS":
        search_labels = [['none', 'no causes'], ['bias', 'abuse'], ['job', 'career'], 'medication', 'relationship', 'alienation']
    else:
        raise NameError(f"{dataset_name} is not a valid dataset name")
    return search_labels