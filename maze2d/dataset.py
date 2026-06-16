import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from typing import Dict, Any

from d4rl.sequence import GoalDataset, Batch


# Bring it into right format
# {'x': trajectories, 'conditions': {idx: value}}
def goal_dataset_collate_fn(batch: Batch) -> Dict[str, Any]:
    x = torch.stack([torch.as_tensor(b.trajectories) for b in batch], dim=0)
    keys = batch[0].conditions.keys()  # GoalDataset: {0, horizon-1}
    conditions = {
        k: torch.stack([torch.as_tensor(b.conditions[k]) for b in batch], dim=0)
        for k in keys
    }
    return {"x": x, "conditions": conditions}


class Maze2dGoalDataModule(pl.LightningDataModule):
    def __init__(self, dataset: GoalDataset, batch_size: int, num_workers: int = 0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=goal_dataset_collate_fn, # type: ignore
        )

