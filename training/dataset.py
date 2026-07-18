"""
Dataset loader for MultiTask-CodePilot.
"""

import json
from torch.utils.data import Dataset

from training.prompts import build_prompt


class MultiTaskDataset(Dataset):

    def __init__(self, dataset_path):

        self.records = []

        with open(dataset_path, "r", encoding="utf-8") as f:
            for line in f:
                self.records.append(json.loads(line))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):

        record = self.records[idx]

        return {
            "text": build_prompt(record)
        }