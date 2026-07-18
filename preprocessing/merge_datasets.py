import json
import random
from pathlib import Path

random.seed(42)

INPUT_FILES = [
    Path("datasets/processed/mbpp_clean.jsonl"),
    Path("datasets/processed/documint_clean.jsonl"),
    Path("datasets/processed/xlcost_clean.jsonl"),
]

OUTPUT_FILE = Path("datasets/processed/unified_dataset.jsonl")


def load_jsonl(file_path):
    records = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def save_jsonl(records, file_path):
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

    merged = []

    print("=" * 70)

    for file in INPUT_FILES:
        records = load_jsonl(file)
        merged.extend(records)
        print(f"{file.name:<25} {len(records):>8}")

    print("-" * 70)
    print(f"{'Total':<25} {len(merged):>8}")

    random.shuffle(merged)

    save_jsonl(merged, OUTPUT_FILE)

    print("\nMerged dataset saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()