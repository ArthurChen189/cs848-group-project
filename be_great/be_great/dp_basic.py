from math import inf
import warnings
import json
import typing as tp
import logging

import dp_transformers
import fsspec
import numpy as np
import pandas as pd

from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments

from be_great.dp import dp_collator
from be_great.dp.dp_collator import DataCollatorDPLLMTGen
from be_great.dp_basic_trainer import DPBasicTrainer
from dp_transformers.arguments import PrivacyArguments
from be_great.great import GReaT
from be_great.great_dataset import GReaTDataset, GReaTDataCollator
from be_great.great_start import (
    GReaTStart,
    CategoricalStart,
    ContinuousStart,
    RandomStart,
    _pad_tokens,
)
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import (
    _array_to_dataframe,
    _get_column_distribution,
    _convert_tokens_to_text,
    _convert_text_to_tabular_data,
    _partial_df_to_promts,
    bcolors,
)


class DPBasic(GReaT):
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
        **train_kwargs,
    ):
        """Initializes GReaT.

        Args:
            llm: HuggingFace checkpoint of a pretrained large language model, used a basis of our model
            experiment_dir:  Directory, where the training checkpoints will be saved
            epochs: Number of epochs to fine-tune the model
            batch_size: Batch size used for fine-tuning
            efficient_finetuning: Indication of fune-tuning method
            train_kwargs: Additional hyperparameters added to the TrainingArguments used by the HuggingFaceLibrary,
             see here the full list of all possible values
             https://huggingface.co/docs/transformers/main/en/main_classes/trainer#transformers.TrainingArguments
        """
        # Load Model and Tokenizer from HuggingFace
        self.efficient_finetuning = efficient_finetuning
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(self.llm)

        if self.efficient_finetuning == "lora":
            # Lazy importing
            try:
                from peft import (
                    LoraConfig,
                    get_peft_model,
                    prepare_model_for_int8_training,
                    TaskType,
                )
            except ImportError:
                raise ImportError(
                    "This function requires the 'peft' package. Please install it with - pip install peft==0.9.0"
                )

            # Define LoRA Config
            lora_config = LoraConfig(
                r=16,  # only training 0.16% of the parameters of the model
                lora_alpha=32,
                target_modules=[
                    "c_attn"
                ],  # this is specific for gpt2 model, to be adapted
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM,  # this is specific for gpt2 model, to be adapted
            )
            # prepare int-8 model for training
            self.model = prepare_model_for_int8_training(self.model)
            # add LoRA adaptor
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Set the training hyperparameters
        self.experiment_dir = experiment_dir
        self.epochs = epochs
        self.batch_size = batch_size
        self.train_hyperparameters = train_kwargs

        # Needed for the sampling process
        self.columns = None
        self.num_cols = None
        self.conditional_col = None
        self.conditional_col_dist = None

    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> DPBasicTrainer:
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
        # data_collator = DataCollatorDPLLMTGen(self.tokenizer) 
        data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(self.tokenizer) 


        great_trainer = DPBasicTrainer(
            model=self.model,
            args=training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            privacy_args=PrivacyArguments(
                per_sample_max_grad_norm=10., 
                target_epsilon=10, 
                # disable_dp=True
                ),
        )

        great_trainer.train()
        return great_trainer

        # Start training
        logging.info("Start training...")
        try:
            great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        finally:
            eps_prv = great_trainer.get_prv_epsilon()
            eps_rdp = great_trainer.get_rdp_epsilon()
            great_trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })
            return great_trainer


   