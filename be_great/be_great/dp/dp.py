import decimal
import logging
import typing as tp
import numpy as np
import pandas as pd
# from opacus import PrivacyEngine
import torch
from be_great.dp.dp_collator import DataCollatorDPLLMTGen
from be_great.dp.dp_trainer import DPLLMTGenTrainer
from be_great.great import GReaT
from be_great.great_dataset import GReaTDataCollator, GReaTDataset
from be_great.great_trainer import GReaTTrainer
from be_great.great_utils import _array_to_dataframe
from transformers import TrainingArguments
from dp_transformers.arguments import PrivacyArguments



class DPLLMTGen(GReaT):

    def __init__(
        self,
        llm: str,
        experiment_dir: str = "trainer_great",
        epochs: int = 100, # No longer used
        batch_size: int = 8,
        efficient_finetuning: str = "",
        stage1_epochs: int = 10,
        stage2_epochs: int = 4,
        stage1_lr: float=1e-4,
        stage2_lr: float=5e-4,
        loss_alpha: float = 0.65,
        loss_beta: float = 0.5,
        loss_lmbda: float = 1,
        device: str = "cuda",
        **train_kwargs,
    ):
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.stage1_lr = stage1_lr
        self.stage2_lr = stage2_lr
        self.loss_alpha = loss_alpha
        self.loss_beta = loss_beta
        self.loss_lmbda = loss_lmbda
        self.device = device
        super().__init__(llm, experiment_dir, epochs, batch_size, efficient_finetuning, **train_kwargs)


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

        trainer = self.fit_format_learning(df, resume_from_checkpoint)
        trainer = self.fit_DP_finetune(df)
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
                    max_num_decimals = abs(col.map(lambda num: decimal.Decimal(str(num)).as_tuple().exponent).min())
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
    
    def fit_DP_finetune(self, df: pd.DataFrame) -> GReaTTrainer:

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
            per_device_train_batch_size=self.batch_size,
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
        # TODO configure privacy args
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
            privacy_args=PrivacyArguments(per_sample_max_grad_norm=1.0, target_epsilon=10.),
        )

        try:
            trainer.train()
        finally:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            trainer.log({
                "final_epsilon_prv": eps_prv,
                "final_epsilon_rdp": eps_rdp
            })
            return trainer

        # # define your components as usual
        # model = self.model
        # optimizer = great_trainer.create_optimizer()
        # data_loader = great_trainer.get_train_dataloader()

        # print(model)
        # print(optimizer)
        # print(data_loader)

        # privacy_engine = PrivacyEngine()
        # model, optimizer, data_loader = privacy_engine.make_private(
        #     module=model,
        #     optimizer=optimizer,
        #     data_loader=data_loader,
        #     noise_multiplier=1.1,
        #     max_grad_norm=1.0,
        # )

        # great_trainer.model = model
        # great_trainer.optimizer = optimizer
        # great_trainer.dp_train_dataloader = data_loader
        
        # # Start training
        # logging.info("Start DP finetuning...")
        # great_trainer.train()
        # return great_trainer
