import torch
from transformers import PreTrainedTokenizer
from typing import Any, List, Union, Dict

from dp_transformers.dp_utils import DataCollatorForPrivateCausalLanguageModeling

class DataCollatorDPLLMTGen(DataCollatorForPrivateCausalLanguageModeling):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer, mlm=False)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = super().__call__(examples)

        batch = self.tokenizer.pad(
            examples,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()
        return batch