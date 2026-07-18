########################################################
# CodePilot AI
#
# utils.py
#
# Helper functions for preprocessing
########################################################



import json
from pathlib import Path


def ensure_directory(directory):

    Path(directory).mkdir(
        parents=True,
        exist_ok=True
    )


def save_jsonl(records, output_file):

    ensure_directory(
        Path(output_file).parent
    )

    with open(output_file, "w", encoding="utf-8") as f:

        for record in records:

            f.write(
                json.dumps(record, ensure_ascii=False)
            )

            f.write("\n")


def print_summary(dataset_name, records, output_file):

    print()
    print("=" * 60)
    print(dataset_name.upper())
    print("=" * 60)

    print(f"Records : {len(records)}")
    print(f"Saved   : {output_file}")

    print("=" * 60)