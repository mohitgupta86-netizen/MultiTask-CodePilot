########################################################
# CodePilot AI
#
# Convert DocuMint -> Unified Dataset
########################################################

from datasets import load_dataset

from utils import save_jsonl
from utils import print_summary


OUTPUT_FILE = "datasets/processed/documint.jsonl"


def build_dataset():

    dataset = load_dataset("documint/DocuMint")

    records = []

    for sample in dataset["train"]:

        record = {

            "task": "documentation",

            "parameters": {

                "language": "python"

            },

            "input": sample["instruction"],

            "output": sample["response"]

        }

        records.append(record)

    save_jsonl(
        records,
        OUTPUT_FILE
    )

    print_summary(
        "DocuMint",
        records,
        OUTPUT_FILE
    )


if __name__ == "__main__":

    build_dataset()