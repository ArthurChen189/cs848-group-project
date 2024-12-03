import torch
from transformers import PreTrainedTokenizer
from typing import Any, List, Union, Dict

from transformers.data import DataCollatorWithPadding

class DataCollatorDPLLMTGen(DataCollatorWithPadding):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)

    def __call__(self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        batch = self.tokenizer.pad(
            examples,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch["labels"] = batch["input_ids"].clone()

        # This is copied from DP-Transformers
        # from dp_transformers.dp_utils import DataCollatorForPrivateCausalLanguageModeling

        # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # since Opacus is not able to deduce the batch size from the input. Here we manually
        # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # but it is constructed in a way that is compatile with Opacus by using expand_as.
        if "position_ids" not in batch:
            input_ids = batch["input_ids"]
            batch["position_ids"] = torch.arange(
                input_ids.shape[1], dtype=torch.long, device=input_ids.device
            ).repeat(input_ids.shape[0], 1)

        return super().__call__(batch)