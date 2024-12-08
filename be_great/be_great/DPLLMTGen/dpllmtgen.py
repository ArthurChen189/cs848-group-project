import decimal
import logging
import typing as tp
import numpy as np
import pandas as pd
import torch

from transformers import TrainingArguments
from be_great.DPLLMTGen.dpllmtgen_trainer_nodp import DPLLMTGenTrainerNoDP
from be_great.DPLLMTGen.dpllmtgen_trainer import DPLLMTGenTrainer
from be_great.dp_collator import DataCollatorDPLLMTGen
from be_great.great_dp import GReaTDP
from be_great.great_dataset import GReaTDataCollator, GReaTDataset
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import _array_to_dataframe


class DPLLMTGen(GReaTDP):

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100, # No longer used
        batch_size: int = 8,
        efficient_finetuning: str = "",
        per_sample_max_grad_norm=1., 
        target_epsilon=1.,
        noise_multiplier=None,
        stage1_epochs: int = 600,
        stage2_epochs: int = 400,
        stage1_lr: float=5e-5,
        stage2_lr: float=2.5e-5,
        loss_alpha: float = 0.65,
        loss_beta: float = 0.1,
        loss_lmbda: float = 1,
        stage2_batch_size: tp.Optional[int] = None,
        use_dp: bool = True,
        device: str = "cuda",
        skip_to_stage2: bool = False,
        **train_kwargs,
    ):
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_lmbda = loss_lmbda
        self.stage2_batch_size = stage2_batch_size
        self.use_dp = use_dp
        self.skip_to_stage2 = skip_to_stage2
        self.device = device
        super().__init__(llm, experiment_dir, epochs, batch_size, efficient_finetuning, per_sample_max_grad_norm=per_sample_max_grad_norm, target_epsilon=target_epsilon, noise_multiplier=noise_multiplier, **train_kwargs)


    def fit(
        self,
        data: tp.Union[pd.DataFrame, np.ndarray],
        column_names: tp.Optional[tp.List[str]] = None,
        conditional_col: tp.Optional[str] = None,
        resume_from_checkpoint: tp.Union[bool, str] = False,
    ) -> GReaTTrainer:

        df = _array_to_dataframe(data, columns=column_names)
        self._update_column_information(df)
        self._update_conditional_information(df, conditional_col)

        if not self.skip_to_stage2:
            trainer = self.fit_format_learning(df, resume_from_checkpoint)
        trainer = self.fit_DP_finetune(df, resume_from_checkpoint)
        return trainer
    
    def fit_format_learning(self, df: pd.DataFrame, resume_from_checkpoint: bool | str = False) -> GReaTTrainer:

        # Generate a random dataframe
        rand_df = pd.DataFrame()
        for col_name in df.columns:
            col = df[col_name]
            if col_name in self.num_cols:
                # DP-LLMTGen paper's requirement is that numbers are in correct range
                minval, maxval = col.min(), col.max()
                all_ints = (col % 1  == 0).all()
                if all_ints:
                    rand_df[col_name] = np.random.randint(minval, maxval+1, size=col.size)
                else:
                    try:
                        max_num_decimals = abs(col.map(lambda num: decimal.Decimal(str(num)).as_tuple().exponent).min())
                    except Exception as e:
                        logging.info("Unable to find max_num_decimals, defaulting to 2 decimal places")
                        max_num_decimals=2
                    rand_df[col_name] = np.random.uniform(minval, maxval, size=col.size).round(max_num_decimals)
            else:
                rand_df[col_name] = np.random.choice(col.unique(), size=col.size)

        # Convert DataFrame into HuggingFace dataset object
        # TODO perform train test split
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(rand_df)
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.stage1_epochs,
            learning_rate=self.stage1_lr,
            per_device_train_batch_size=self.batch_size,
            **self.train_hyperparameters,
        )
        great_trainer = GReaTTrainer(
            self.model,
            training_args,
            train_dataset=great_ds,
            tokenizer=self.tokenizer,
            data_collator=GReaTDataCollator(self.tokenizer),
        )
        
        # Start training
        logging.info("Start format learning...")
        great_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        return great_trainer
    
    def fit_DP_finetune(self, df: pd.DataFrame, resume_from_checkpoint: bool | str = False) -> GReaTTrainer:

        # Convert DataFrame into HuggingFace dataset object
        logging.info("Convert data into HuggingFace dataset object...")
        great_ds = GReaTDataset.from_pandas(df)
        great_ds.set_tokenizer(self.tokenizer)

        # Set training hyperparameters
        logging.info("Create GReaT Trainer...")
        training_args = TrainingArguments(
            self.experiment_dir,
            num_train_epochs=self.stage2_epochs,
            learning_rate=self.stage2_lr,
            per_device_train_batch_size=self.stage2_batch_size if self.stage2_batch_size else self.batch_size,
            remove_unused_columns=False,
            save_safetensors=False, # See https://github.com/microsoft/dp-transformers/issues/51
            **self.train_hyperparameters,
        )
        
        # Find format token ids
        format_token_ids = []
        for col_name in df:
            format_token_ids.append(torch.tensor(self.tokenizer.encode(f"{col_name} is"), device=self.device))
            format_token_ids.append(torch.tensor(self.tokenizer.encode(f", {col_name} is"), device=self.device))


        # Find numerical token ids
        numerical_token_ids = []
        for num in range(10):
            numerical_token_ids.append(torch.tensor(self.tokenizer.encode(f" {num}"), device=self.device))
            numerical_token_ids.append(torch.tensor(self.tokenizer.encode(f"{num}"), device=self.device))
        numerical_token_ids.append(torch.tensor(self.tokenizer.encode("."), device=self.device))

        # Create a trainer first to get the default optimzer and dataloader
        data_collator = DataCollatorDPLLMTGen(self.tokenizer) 
        if self.use_dp:
            trainer = DPLLMTGenTrainer(
                format_token_ids,
                numerical_token_ids,
                alpha=self.loss_alpha,
                beta=self.loss_beta,
                lmbda=self.loss_lmbda,
                model=self.model,
                args=training_args,
                train_dataset=great_ds,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                max_physical_batch_size=self.max_physical_batch_size,
                privacy_args=self.privacy_args,
            )
        else:
            trainer = DPLLMTGenTrainerNoDP(
                format_token_ids,
                numerical_token_ids,
                alpha=self.loss_alpha,
                beta=self.loss_beta,
                lmbda=self.loss_lmbda,
                model=self.model,
                args=training_args,
                train_dataset=great_ds,
                tokenizer=self.tokenizer,
                data_collator=data_collator,
            )

        # trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        trainer.train()

        if self.use_dp:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            privacy_stats = {
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            }
            print(privacy_stats)
            trainer.log(privacy_stats)
        return trainer
