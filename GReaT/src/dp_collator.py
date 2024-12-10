import torch
from transformers import PreTrainedTokenizer
from typing import List, Union, Dict

from dp_transformers.dp_utils import DataCollatorForPrivateCausalLanguageModeling

class DataCollatorDPLLMTGen(DataCollatorForPrivateCausalLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)
        batch["labels"] = batch["input_ids"].clone()
        return batch