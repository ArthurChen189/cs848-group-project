import random
from typing import Any, Dict, Union
import numpy as np

import torch
from torch.utils.data import DataLoader
from dp_transformers.dp_utils import OpacusDPTrainer



def _seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class GReaTDPTrainer(OpacusDPTrainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        return super().training_step(model, inputs)

    def get_train_dataloader(self) -> DataLoader:    
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use the author-level sampler from dp_transformers.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        return DataLoader(
            self.train_dataset,
            batch_sampler=self._get_train_sampler(),
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

