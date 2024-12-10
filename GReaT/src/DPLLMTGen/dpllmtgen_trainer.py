from typing import List
from src.DPLLMTGen.dpllmtgen_loss import DPLLMTGenLoss
from src.great_dp_trainer import GReaTDPTrainer

class DPLLMTGenTrainer(GReaTDPTrainer):
    def __init__(
            self, 
            format_token_ids: List[int], 
            numerical_token_ids: List[int], 
            alpha: float,
            beta: float, 
            lmbda: float, 
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.dpllmtgen_loss = DPLLMTGenLoss(
            format_token_ids=format_token_ids,
            numerical_token_ids=numerical_token_ids,
            processing_class=self.processing_class,
            alpha = alpha,
            beta = beta,
            lmbda = lmbda,
        )

    def compute_loss(self, model, inputs, return_outputs = False, num_items_in_batch = None):
        return self.dpllmtgen_loss.compute_loss(model, inputs)