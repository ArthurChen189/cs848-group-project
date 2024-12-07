from dp_transformers.dp_utils import OpacusDPTrainer, DataCollatorForPrivateCausalLanguageModeling
import torch
from transformers import Seq2SeqTrainer

from rtf_datacollator import RelationalDataCollator

class Seq2SeqTrainerDP(Seq2SeqTrainer, OpacusDPTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Seq2SeqDataCollator(RelationalDataCollator):
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors)
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)
        return batch