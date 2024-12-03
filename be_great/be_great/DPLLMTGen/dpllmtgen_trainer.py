from typing import Any, Dict, List, Union
import dp_transformers
import numpy as np
import torch
from torch.utils.data import DataLoader

class DPLLMTGenTrainer(dp_transformers.dp_utils.OpacusDPTrainer):
    def __init__(
            self, 
            format_token_ids, 
            numerical_token_ids, 
            alpha:float = 0.65,
            beta:float = .1, 
            lmbda:float = 1, 
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.format_token_ids = format_token_ids
        self.numerical_token_ids = numerical_token_ids
        self.tokenizer = kwargs["tokenizer"]
        self.alpha = alpha
        self.beta = beta
        self.lmbda = lmbda

        self.dp_train_dataloader = None

    def get_train_dataloader(self) -> DataLoader:
        return super().get_train_dataloader()
        if self.dp_train_dataloader is None:
            return super().get_train_dataloader()
        else:
            return self.dp_train_dataloader
    
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]], num_items_in_batch=None) -> torch.Tensor:
        return super().training_step(model, inputs)


    
    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        # loss = super().compute_loss(model, inputs)
        # print(loss)
        # return loss
        input_ids = inputs.get("input_ids")
        # outputs = model(input_ids)
        outputs = model(**inputs)

        batch_size = input_ids.size(0)

        # Shift so that tokens < n predict n
        shift_labels = input_ids[..., 1:].contiguous()
        shift_logits = outputs.logits[..., :-1, :].contiguous()

        
        input_format_token_idx = self._find_tokens(input_ids, self.format_token_ids)

        # Borrow the standard cross entropy loss but give different weight to format vs tabular tokens
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        # loss_fn = torch.nn.CrossEntropyLoss()


        # Calculate per token loss
        wce_losses = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # print(shift_logits.view(-1, shift_logits.size(-1)).shape)
        # print(loss.shape)

        # print(wce_losses)
        # return wce_losses

        # reshape to per token loss per sample
        wce_losses = wce_losses.view(shift_logits.size(0), shift_logits.size(1))
        # print(wce_loss.shape)

        # Weigh loss based on token type
        loss_format_token_bitmap = input_format_token_idx[...,1:]
        wce_losses[loss_format_token_bitmap] *= (1 - self.alpha)
        wce_losses[~loss_format_token_bitmap] *= self.alpha
        # wce_losses = wce_losses.sum(axis=1)
        wce_losses = wce_losses.mean(axis=1)
        # print(loss_format_token_bitmap.sum() / loss_format_token_bitmap.numel())
        # print(wce_losses.size())
        # print(wce_losses)
        # return wce_losses

        # Find Numerical Understanding Loss
        highest_prob_predictions = torch.argmax(shift_logits, dim=-1)

        
        label_num_token_idx = self._find_tokens(shift_labels, self.numerical_token_ids)
        pred_num_token_idx = self._find_tokens(highest_prob_predictions, self.numerical_token_ids)

        nu_losses = torch.zeros(batch_size, device=input_ids.device)
        # print(self.processing_class.decode(shift_labels[0]))
        # print(self.processing_class.decode(highest_prob_predictions[0]))
        for sample in range(batch_size):
            # print("***")
            # print((shift_labels[sample])[label_num_token_idx[sample]])
            # print((highest_prob_predictions[sample])[pred_num_token_idx[sample]])
            label_num_strs = self.processing_class.decode((shift_labels[sample])[label_num_token_idx[sample]]).strip().split(" ")
            pred_num_strs = self.processing_class.decode((highest_prob_predictions[sample])[pred_num_token_idx[sample]]).strip().split(" ")
            # print(label_num_strs)
            # print(pred_num_strs)
            diffs = []
            failed_count = abs(len(label_num_strs) - len(pred_num_strs))
            for label_num_str, pred_num_str in zip(label_num_strs, pred_num_strs):
                label_num = decode_str_to_num(label_num_str)
                pred_num = decode_str_to_num(pred_num_str)
                if label_num and pred_num:
                    diffs.append(label_num - pred_num)
                else:
                    failed_count += 1
            diffs = np.array(diffs)
            sqrd_errors = 0.5 * ((diffs / self.lmbda) ** 2)
            sample_nu_loss = sqrd_errors.sum() + failed_count
            nu_losses[sample] = sample_nu_loss / len(label_num_strs)
                
        loss = (wce_losses + self.beta * nu_losses).mean()
        # print(f"loss: {loss} wce_losses: {wce_losses.mean()} nu_losses {self.beta * nu_losses.mean()}")
        # loss = wce_losses.mean()
        return loss
    

    def _find_tokens(self, input_ids, token_list: List[torch.tensor]):
        # print(input_ids.size())
        # for sample in input_ids:
        #     for input_id in sample:
        #         print(input_id, f"|{self.processing_class.decode(input_id)}|")
        #     print(self.processing_class.decode(sample))

        m = input_ids.size(1) # prob not shift labels
        format_token_idx = torch.zeros(input_ids.size(), dtype=torch.bool, device=input_ids.device)

        # self.format_token_ids = [torch.tensor(self.tokenizer.encode(f"petal width is"), device="cuda")]
        for format_pattern in token_list:
            n = len(format_pattern)
            slices = torch.stack([input_ids.narrow(1, i, m-n+1) for i in range(n)], dim=-1)
            # print(slices.size())
            # print(slices == format_pattern)
            # print(slices)
            matches = torch.all(slices == format_pattern, dim=-1)
            # print(matches)
            # print(format_pattern, self.processing_class.decode(format_pattern))
            padded_matches = torch.zeros(input_ids.size(), dtype=torch.bool, device=input_ids.device)
            padded_matches[...,:m-n+1] = matches
            for i in range(n-1):
                padded_matches |= padded_matches.roll(1)

            format_token_idx |= padded_matches
        # Sanity check to make sure that the format token positions are correct
        # print(format_token_idx)
        # for sample, idx in zip(input_ids, format_token_idx):
        #     print(self.processing_class.decode(sample))
        #     print(self.processing_class.decode(sample[idx]))
        return format_token_idx
    
    
def decode_str_to_num(s):
    try:
        return float(s)
    except ValueError:
        return False
        
def find_contiguous_segments(tensor: torch.tensor) -> List[List[int]]:
    # Unused
    print(tensor.size())
    non_matching = (tensor[:-1] != tensor[1:]) < 0  # find end segment mask
    # Note: instead of checking for "0" diff we simply check for non-equality

    size = tensor.size()[0]

    # index range so we can return start/end indices
    index_range = torch.arange(0, size, device=tensor.device).reshape(-1,1)

    ends = index_range[1:][non_matching]  # end indices, as in the diagram above

    # print(ends)
    # print(torch.tensor([[0]], device=tensor.device).size())
    # print(ends.size())
    starts = torch.concat((torch.tensor([[0]], device=tensor.device), ends))  # add initial index
    ends = torch.concat((ends, torch.tensor([[size]], device=tensor.device)))  # add end index

    print(starts.shape)
    print(ends.shape)
    indices_tensor = torch.concat((starts, ends), dim=1)

    # Note: much faster than looping!
    indices_list = indices_tensor.tolist()

    return indices_list