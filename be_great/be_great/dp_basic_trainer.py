import random
from typing import Any, Dict, Union
import numpy as np

import opacus.optimizers
import torch
from torch.utils.data import DataLoader

import opacus
from transformers import Trainer, training_args
import dp_transformers



def _seed_worker(_):
    """
    Helper function to set worker seed during Dataloader initialization.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.cuda.manual_seed_all(worker_seed)


class DPBasicTrainer(dp_transformers.dp_utils.OpacusDPTrainer):
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        return super().training_step(model, inputs)
    
    # def create_optimizer(self):
    #     _ = super().create_optimizer()

    #     if self.args.parallel_mode == training_args.ParallelMode.DISTRIBUTED:
    #         optimizer_generator = opacus.optimizers.DistributedDPOptimizer
    #     else:
    #         optimizer_generator = opacus.optimizers.DPOptimizer
    #         # optimizer_generator = CustomDPOptimizer

    #     self.optimizer = optimizer_generator(
    #         optimizer=self.optimizer,
    #         noise_multiplier=self.privacy_args.noise_multiplier,
    #         max_grad_norm=self.privacy_args.per_sample_max_grad_norm,
    #         expected_batch_size=self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps,
    #     )

    #     return self.optimizer

    def get_train_dataloader(self) -> DataLoader:
        return super().get_train_dataloader()
    
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        data_collator = self.data_collator
        train_dataset = (
            self.train_dataset
        )  # self._remove_unused_columns(self.train_dataset, description="training")
        train_sampler = self._get_train_sampler()

        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=_seed_worker,
        )
