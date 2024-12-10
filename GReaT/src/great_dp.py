import typing as tp
import logging

import numpy as np
import pandas as pd

# See https://github.com/microsoft/dp-transformers/blob/main/README.md
from dp_transformers.grad_sample.transformers import conv_1d 

from transformers import  TrainingArguments

from src.dp_collator import DataCollatorDPLLMTGen
from src.great_dp_trainer import GReaTDPTrainer
from dp_transformers.arguments import PrivacyArguments
from src.great import GReaT
from src.great_dataset import GReaTDataset
from src.great_utils import (
    _array_to_dataframe,
)


class GReaTDP(GReaT):
    """GReaT Class

    The GReaT class handles the whole generation flow. It is used to fine-tune a large language model for tabular data,
    and to sample synthetic tabular data.

    Attributes:
        llm (str): HuggingFace checkpoint of a pretrained large language model, used a basis of our model
        tokenizer (AutoTokenizer): Tokenizer, automatically downloaded from llm-checkpoint
        model (AutoModelForCausalLM): Large language model, automatically downloaded from llm-checkpoint
        experiment_dir (str): Directory, where the training checkpoints will be saved
        epochs (int): Number of epochs to fine-tune the model
        batch_size (int): Batch size used for fine-tuning
        train_hyperparameters (dict): Additional hyperparameters added to the TrainingArguments used by the
         HuggingFaceLibrary, see here the full list of all possible values
         https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        columns (list): List of all features/columns of the tabular dataset
        num_cols (list): List of all numerical features/columns of the tabular dataset
        conditional_col (str): Name of a feature/column on which the sampling can be conditioned
        conditional_col_dist (dict | list): Distribution of the feature/column specified by condtional_col
    """

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100,
        batch_size: int = 8,
        efficient_finetuning: str = "",
        per_sample_max_grad_norm=1., 
        target_epsilon=1., 
        target_delta=None,
        noise_multiplier=None,
        max_physical_batch_size:tp.Optional[int]=None,
        **train_kwargs,
    ):
        self.per_sample_max_grad_norm = per_sample_max_grad_norm
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.noise_multiplier = noise_multiplier
        self.privacy_args = PrivacyArguments(
            per_sample_max_grad_norm=per_sample_max_grad_norm,
            target_epsilon=target_epsilon if noise_multiplier is None else None,
            target_delta=target_delta,
            noise_multiplier=noise_multiplier
        )
        self.max_physical_batch_size = max_physical_batch_size if max_physical_batch_size else batch_size
        super().__init__(llm, experiment_dir, epochs, batch_size, efficient_finetuning, **train_kwargs)

    # @property
    # def privacy_args(self) -> PrivacyArguments:
    #     """
    #     Returns privacy arguments
    #     """
    #     if self.noise_multiplier:
    #         self.target_epsilon = None
    #     return PrivacyArguments(
    #             per_sample_max_grad_norm=self.per_sample_max_grad_norm, 
    #             target_epsilon=self.target_epsilon, 
    #             noise_multiplier=self.noise_multiplier,
    #             # disable_dp=True
    #             ),

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> GReaTDPTrainer:
        """Fine-tune GReaT using tabular data.

        Args:
            data: Pandas DataFrame or Numpy Array that contains the tabular data
            column_names: If data is Numpy Array, the feature names have to be defined. If data is Pandas
            DataFrame, the value is ignored
            conditional_col: If given, the distribution of this column is saved and used as a starting
            point for the generation process later. If None, the last column is considered as conditional feature
            resume_from_checkpoint: If True, resumes training from the latest checkpoint in the experiment_dir.
            If path, resumes the training from the given checkpoint (has to be a valid HuggingFace checkpoint!)

        Returns:
            GReaTTrainer used for the fine-tuning process
        """
        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.epochs,
            per_device_train_batch_size=self.batch_size,
            remove_unused_columns=False,
            **self.train_hyperparameters,
        )
        data_collator = DataCollatorDPLLMTGen(self.tokenizer) 


        great_trainer = GReaTDPTrainer(
            max_physical_batch_size=self.max_physical_batch_size,
            model=self.model,
            args=training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            privacy_args=self.privacy_args,
        )

        # Start training
        logging.info("Start training...")
        try:
            great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        finally:
            eps_prv = great_trainer.get_prv_epsilon()
            eps_rdp = great_trainer.get_rdp_epsilon()
            privacy_stats = {
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            }
            print(privacy_stats)
            great_trainer.log(privacy_stats)
            return great_trainer


   