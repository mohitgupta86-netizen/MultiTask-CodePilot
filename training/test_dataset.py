from transformers import AutoTokenizer

from training.config import MODEL_NAME, DATASET_PATH
from training.dataset import MultiTaskDataset


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Qwen requires a pad token
tokenizer.pad_token = tokenizer.eos_token

dataset = MultiTaskDataset(DATASET_PATH, tokenizer)

print(f"Dataset Size : {len(dataset)}")

sample = dataset[0]

print(sample.keys())

print(sample["input_ids"].shape)
print(sample["attention_mask"].shape)
print(sample["labels"].shape)