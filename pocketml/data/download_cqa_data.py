import os
import requests

def download(target_dir):
    os.makedirs(target_dir)
    print("Downloading training data...")
    r_train = requests.get("https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl")
    print("Downloading validation data...")
    r_valid = requests.get("https://s3.amazonaws.com/commensenseqa/dev_rand_split.jsonl")
    print("Downloading test data...")
    r_test = requests.get("https://s3.amazonaws.com/commensenseqa/test_rand_split_no_answers.jsonl")
    print("Downloading dictionary...")
    r_dict = requests.get("https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt")

    data_requests = [r_train, r_valid, r_test, r_dict]
    target_files = ["train.jsonl", "valid.jsonl", "test.jsonl", "dict.txt"]
    for filename, request in zip(target_files, data_requests):
        with open(f"{target_dir}/{filename}", "w", encoding="utf-8") as fp:
            fp.write(request.text)

    print(f"Downloaded data to {target_dir}")
