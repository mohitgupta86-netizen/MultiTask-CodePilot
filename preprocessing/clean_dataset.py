import json
from pathlib import Path

# ==============================
# Configuration
# ==============================

MAX_INPUT_CHARS = 20000
MAX_OUTPUT_CHARS = 10000

DATASETS = [
    Path("datasets/processed/mbpp.jsonl"),
    Path("datasets/processed/documint.jsonl"),
    Path("datasets/processed/xlcost.jsonl"),
]


# ==============================
# Helper Functions
# ==============================

def load_jsonl(file_path):
    records = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    return records


def save_jsonl(records, output_path):

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ==============================
# Cleaning Logic
# ==============================

def clean_dataset(file_path):

    print("=" * 70)
    print(f"Cleaning : {file_path.name}")
    print("=" * 70)

    records = load_jsonl(file_path)

    cleaned = []

    seen = set()

    removed_empty_inputs = 0
    removed_empty_outputs = 0
    removed_duplicates = 0
    removed_long_samples = 0

    for record in records:

        task = record.get("task", "")

        parameters = json.dumps(
            record.get("parameters", {}),
            sort_keys=True
        )

        inp = record.get("input", "")
        out = record.get("output", "")

        # ----------------------------
        # Empty input
        # ----------------------------

        if not inp.strip():
            removed_empty_inputs += 1
            continue

        # ----------------------------
        # Empty output
        # ----------------------------

        if not out.strip():
            removed_empty_outputs += 1
            continue

        # ----------------------------
        # Very long samples
        # ----------------------------

        if len(inp) > MAX_INPUT_CHARS:
            removed_long_samples += 1
            continue

        if len(out) > MAX_OUTPUT_CHARS:
            removed_long_samples += 1
            continue

        # ----------------------------
        # Duplicate removal
        # ----------------------------

        key = (
            task,
            parameters,
            inp,
            out
        )

        if key in seen:
            removed_duplicates += 1
            continue

        seen.add(key)
        cleaned.append(record)

    output_file = file_path.with_name(
        file_path.stem + "_clean.jsonl"
    )

    save_jsonl(cleaned, output_file)

    print(f"Original Samples      : {len(records)}")
    print(f"Removed Empty Inputs  : {removed_empty_inputs}")
    print(f"Removed Empty Outputs : {removed_empty_outputs}")
    print(f"Removed Duplicates    : {removed_duplicates}")
    print(f"Removed Long Samples  : {removed_long_samples}")
    print(f"Final Samples         : {len(cleaned)}")
    print(f"Saved To              : {output_file}")
    print()


# ==============================
# Main
# ==============================

def main():

    for dataset in DATASETS:

        if dataset.exists():
            clean_dataset(dataset)
        else:
            print(f"Missing: {dataset}")


if __name__ == "__main__":
    main()