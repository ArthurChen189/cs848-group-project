import random
from typing import Any, Dict, Union
import numpy as np

import torch
from torch.utils.data import DataLoader
from opacus.utils.batch_memory_manager import BatchSplittingSampler
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
    def __init__(self, max_physical_batch_size=8, **kwargs):
        self.max_physical_batch_size = max_physical_batch_size
        super().__init__(**kwargs)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        return super().training_step(model, inputs)
    
    @property
    def sampling_probability(self) -> float:
        return self.max_physical_batch_size * self.train_args.world_size * \
            self.train_args.gradient_accumulation_steps / len(self.author_mapping)

    def get_train_dataloader(self) -> DataLoader:    
        """
        Returns the training :class:`~torch.utils.data.DataLoader`.

        Will use the author-level sampler from dp_transformers.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        data_sampler = self._get_train_sampler()

        # Uses the BatchSplittingSampler to simulate larger batch sizes
        self.create_optimizer() # Hack to make sure that the optimizer is initialized
        data_sampler.author_sampler = BatchSplittingSampler(
            sampler=data_sampler.author_sampler,
            max_batch_size=self.max_physical_batch_size,
            optimizer=self.optimizer
        )

        data_loader = DataLoader(
            self.train_dataset,
            batch_sampler=data_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )

        return data_loader

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Modified to support initializing the optimizer in get_train_dataloader()

        """
        # self.create_optimizer()
        # if IS_SAGEMAKER_MP_POST_1_10 and smp.state.cfg.fp16:
        #     # If smp >= 1.10 and fp16 is enabled, we unwrap the optimizer
        #     optimizer = self.optimizer.optimizer
        # else:
        #     optimizer = self.optimizer
        optimizer = self.optimizer
        self.create_scheduler(num_training_steps=num_training_steps, optimizer=optimizer)