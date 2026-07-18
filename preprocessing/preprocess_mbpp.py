########################################################
# CodePilot AI
#
# Convert MBPP -> Unified Dataset
########################################################

from datasets import load_dataset

from utils import save_jsonl
from utils import print_summary


OUTPUT_FILE = "datasets/processed/mbpp.jsonl"


def build_dataset():

    dataset = load_dataset("mbpp")

    records = []

    for sample in dataset["train"]:

        record = {

            "task": "code_generation",

            "parameters": {

                "language": "python"

            },

            "input": sample["text"],

            "output": sample["code"]

        }

        records.append(record)

    save_jsonl(
        records,
        OUTPUT_FILE
    )

    print_summary(
        "MBPP",
        records,
        OUTPUT_FILE
    )


if __name__ == "__main__":

    build_dataset()