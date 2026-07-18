# Preprocessing

This module prepares all datasets for multitask fine-tuning.

## Pipeline

```
Raw Dataset
    ↓
Preprocess
    ↓
Validate
    ↓
Clean
    ↓
Validate
    ↓
Merge
```

## Scripts

| Script | Purpose |
|---------|---------|
| `preprocess_mbpp.py` | Converts MBPP into the unified training format. |
| `preprocess_documint.py` | Converts DocuMint into the unified training format. |
| `preprocess_xlcost.py` | Creates Python ↔ Java translation pairs from XLCoST. |
| `validate_dataset.py` | Performs quality checks such as empty records, duplicates, task distribution, and length statistics. |
| `clean_dataset.py` | Removes empty records, duplicates, and excessively long samples. |
| `merge_datasets.py` | Merges all cleaned datasets into a single training dataset. |

## Output

The preprocessing pipeline produces:

```
datasets/
└── processed/
    ├── mbpp_clean.jsonl
    ├── documint_clean.jsonl
    ├── xlcost_clean.jsonl
    └── unified_dataset.jsonl
```

The `unified_dataset.jsonl` file is used as the input for the fine-tuning pipeline.