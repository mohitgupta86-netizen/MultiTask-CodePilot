from pathlib import Path

BASE = Path(
    "datasets/raw/xlcost/XLCoST_data/XLCoST_data/generation/pair_data_tok_full_desc"
)

python_map = open(
    BASE/"Python-desc"/"train-Python-map.jsonl",
    encoding="utf-8"
).read().splitlines()

java_map = open(
    BASE/"Java-desc"/"train-Java-map.jsonl",
    encoding="utf-8"
).read().splitlines()

print("Python first 5")
print(python_map[:5])

print()

print("Java first 5")
print(java_map[:5])