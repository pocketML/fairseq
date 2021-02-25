import argparse
from fairseq import options
from pocketml.data.data_util import get_dataset_path

def create_parser():
    parser = argparse.ArgumentParser()
    pipeline_tasks = [
        "finetune", "prune", "quantize"
    ]

    finetune_tasks = [
        "commonsense_qa", "glue", "squad"
    ]

    parser.add_argument("task", choices=pipeline_tasks, nargs="+")
    parser.add_argument("--finetune-before", "-ftb", choices=finetune_tasks)
    parser.add_argument("--finetune-during", "-ftd", choices=finetune_tasks)
    parser.add_argument("--finetune-after", "-fta", choices=finetune_tasks)
    parser.add_argument("--config", "-config", required=True)
    return parser

def parse_roberta_args(parser):
    args = parser.parse_args()

    input_args = load_config_file(args.config)

    try:
        dataset = get_dataset_path(args.finetune_before)
    except KeyError:
        print(f"Error: Task '{args.finetune_before}' is not valid.")
        exit(0)

    input_args.append(dataset)
    input_args.extend(["--task", args.finetune_before])

    roberta_parser = options.get_training_parser()
    return options.parse_args_and_arch(roberta_parser, input_args=input_args)

def load_config_file(filename):
    with open(filename, encoding="utf-8") as fp:
        args = []
        for line in fp:
            stripped = line.strip()
            args.extend(stripped.split(" "))
        return args
