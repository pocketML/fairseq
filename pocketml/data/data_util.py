import os
from pocketml.data import download_cqa_data

TASK_DATASET_PATHS = {
    "commonsense_qa": "data/CommonsenseQA"
}

DATASET_DOWNLOAD_FUNCS = {
    "commonsense_qa": download_cqa_data
}

def dataset_exists(path):
    return os.path.exists(path)

def download_data(task, path):
    DATASET_DOWNLOAD_FUNCS[task].download(path)

def get_dataset_path(task):
    path = TASK_DATASET_PATHS[task]
    if not dataset_exists(path):
        download_data(task, path)

    return path
