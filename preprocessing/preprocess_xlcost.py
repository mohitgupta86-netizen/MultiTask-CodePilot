import json
from pathlib import Path

BASE = Path(
    "datasets/raw/xlcost/XLCoST_data/XLCoST_data/generation/pair_data_tok_full_desc"
)

OUTPUT_FILE = Path("datasets/processed/xlcost.jsonl")


def read_lines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def build_translation_dataset():

    python_code = read_lines(
        BASE / "Python-desc" / "train-Python-desc-tok.py"
    )

    java_code = read_lines(
        BASE / "Java-desc" / "train-Java-desc-tok.java"
    )

    python_ids = read_lines(
        BASE / "Python-desc" / "train-Python-map.jsonl"
    )

    java_ids = read_lines(
        BASE / "Java-desc" / "train-Java-map.jsonl"
    )

    assert len(python_ids) == len(python_code)
    assert len(java_ids) == len(java_code)

    python_dict = dict(zip(python_ids, python_code))
    java_dict = dict(zip(java_ids, java_code))

    common_ids = sorted(
        set(python_dict.keys()) & set(java_dict.keys()),
        key=int
    )

    print(f"Python samples : {len(python_dict)}")
    print(f"Java samples   : {len(java_dict)}")
    print(f"Common samples : {len(common_ids)}")

    records = []

    for pid in common_ids:

        py = python_dict[pid]
        java = java_dict[pid]

        records.append({
            "task": "translation",
            "parameters": {
                "source_language": "python",
                "target_language": "java"
            },
            "input": py,
            "output": java
        })

        records.append({
            "task": "translation",
            "parameters": {
                "source_language": "java",
                "target_language": "python"
            },
            "input": java,
            "output": py
        })

    return records


def save_jsonl(records, output_file):

    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():

    records = build_translation_dataset()

    save_jsonl(records, OUTPUT_FILE)

    print(f"Created {len(records)} samples")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()