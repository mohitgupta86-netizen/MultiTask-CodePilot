########################################################
# CodePilot AI
#
# analyse_documint.py
########################################################

from datasets import load_dataset

print("=" * 70)
print("LOADING DOCUMINT")
print("=" * 70)

dataset = load_dataset("documint/DocuMint")

print()
print("Dataset Loaded Successfully")
print()

print("=" * 70)
print("DATASET SPLITS")
print("=" * 70)

print(dataset)

print()

print("=" * 70)
print("NUMBER OF SAMPLES")
print("=" * 70)

for split in dataset.keys():
    print(f"{split:<15}: {len(dataset[split])}")

print()

print("=" * 70)
print("COLUMN NAMES")
print("=" * 70)

print(dataset["train"].column_names)

print()

print("=" * 70)
print("FIRST RECORD")
print("=" * 70)

sample = dataset["train"][0]

for key, value in sample.items():

    print()
    print("-" * 60)
    print(key.upper())
    print("-" * 60)
    print(value)