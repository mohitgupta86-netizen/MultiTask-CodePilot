"""
Dataset loader for MultiTask-CodePilot.
"""

from datasets import load_dataset

from training.prompts import build_prompt


def load_training_dataset(dataset_path):
    """
    Load JSONL dataset and convert each record into a training prompt.
    """

    dataset = load_dataset(
        "json",
        data_files=str(dataset_path),
        split="train",
    )

    print(f"Loaded {len(dataset):,} training samples.")

    dataset = dataset.map(
        lambda record: {"text": build_prompt(record)},
        remove_columns=dataset.column_names,
    )

    return dataset