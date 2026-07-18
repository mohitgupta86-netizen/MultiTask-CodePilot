import json
from pathlib import Path
from collections import Counter


def load_jsonl(file_path):
    """Load all records from a JSONL file."""
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def validate_dataset(file_path):

    print("=" * 70)
    print(f"Dataset : {file_path.name}")
    print("=" * 70)

    records = load_jsonl(file_path)

    print(f"Total Samples : {len(records)}")

    empty_inputs = 0
    empty_outputs = 0
    duplicate_records = 0

    task_counter = Counter()
    translation_counter = Counter()

    seen = set()

    input_lengths = []
    output_lengths = []

    for record in records:

        task = record.get("task", "unknown")
        task_counter[task] += 1

        inp = record.get("input", "")
        out = record.get("output", "")

        if not inp.strip():
            empty_inputs += 1

        if not out.strip():
            empty_outputs += 1

        key = (
            task,
            json.dumps(record.get("parameters", {}), sort_keys=True),
            inp,
            out,
        )

        if key in seen:
            duplicate_records += 1
        else:
            seen.add(key)

        input_lengths.append(len(inp))
        output_lengths.append(len(out))

        if task == "translation":
            params = record.get("parameters", {})
            src = params.get("source_language", "?")
            tgt = params.get("target_language", "?")
            translation_counter[f"{src} -> {tgt}"] += 1

    print("\nTask Distribution")
    print("-" * 30)

    for task, count in task_counter.items():
        print(f"{task:20} {count}")

    if translation_counter:

        print("\nTranslation Direction")
        print("-" * 30)

        for direction, count in translation_counter.items():
            print(f"{direction:20} {count}")

    print("\nQuality Checks")
    print("-" * 30)

    print(f"Empty Inputs      : {empty_inputs}")
    print(f"Empty Outputs     : {empty_outputs}")
    print(f"Duplicate Records : {duplicate_records}")

    print("\nLength Statistics")
    print("-" * 30)

    print(f"Average Input Length  : {sum(input_lengths)/len(input_lengths):.2f}")
    print(f"Average Output Length : {sum(output_lengths)/len(output_lengths):.2f}")

    print(f"Maximum Input Length  : {max(input_lengths)}")
    print(f"Maximum Output Length : {max(output_lengths)}")

    print()


if __name__ == "__main__":

    DATASETS = [
        Path("datasets/processed/unified_dataset.jsonl"),
        # Path("datasets/processed/documint_clean.jsonl"),
        # Path("datasets/processed/xlcost_clean.jsonl"),
    ]

    for dataset in DATASETS:

        if dataset.exists():
            validate_dataset(dataset)
        else:
            print(f"Missing: {dataset}")