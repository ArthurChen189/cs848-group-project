import os
import pandas as pd
from pathlib import Path
import argparse
from src import GReaT, GReaTDP
from src.DPLLMTGen import DPLLMTGen

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, required=True, choices=["GReaT", "DP-GReaT", "DP-LLMTGen", "LLMTGen"], help="Name of the tabular data synthesizer")
    parser.add_argument("--model", type=str, default="gpt2", help="Name or path to the transfoormer model")
    parser.add_argument("--train_df", type=str, required=True, help="Path to the training dataset csv")
    parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
    parser.add_argument("--save_steps", type=int, default=2000, help="Number of steps before saving the model")
    parser.add_argument("--logging_steps", type=int, default=20, help="Number of steps between each outputted log")
    parser.add_argument("--output_dir", type=str, default="privacy_checkpoints", help="Directory to save the models")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    # Privacy
    parser.add_argument("--per_sample_max_grad_norm", type=float, default=1., help="Privacy parameter")
    parser.add_argument("--target_epsilon", type=float, default=1., help="Privacy parameter")
    parser.add_argument("--target_delta", type=float, default=None, help="Privacy parameter")
    parser.add_argument("--max_physical_batch_size", type=int, default=None, help="Max physical batch size during DP training")
    # PEFT and quantization
    parser.add_argument("--efficient_finetuning", type=str, default=None, choices=[None, "lora"], help="Whether to use PEFT techniques, currently only LoRa is supported")
    parser.add_argument("--use_8bit_quantization", action='store_true', help="Whether to perform 8 bit quantization of the model")
    # DP-LLMTGen
    parser.add_argument("--stage1_epochs", type=int, default=5, help="DP-LLMTGen parameter")
    parser.add_argument("--stage2_epochs", type=int, default=2, help="DP-LLMTGen parameter")
    parser.add_argument("--stage1_lr", type=float, default=1e-4, help="DP-LLMTGen parameter")
    parser.add_argument("--stage2_lr", type=float, default=5e-4, help="DP-LLMTGen parameter")
    parser.add_argument("--loss_alpha", type=float, default=0.65, help="DP-LLMTGen parameter")
    parser.add_argument("--loss_beta", type=float, default=0.1, help="DP-LLMTGen parameter")
    parser.add_argument("--loss_lmbda", type=float, default=1.0, help="DP-LLMTGen parameter")
    return parser.parse_args()

def main(args):
    # get the name of the output directory
    dataset_name = args.train_df.split("/")[-2]
    output_dir = Path(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    # Load the data
    df = pd.read_csv(args.train_df)

    # Create the synthesizer object
    match args.name:
        case "GReaT":
            synthesizer = GReaT(
                args.model,
                epochs=args.epochs,
                save_steps=args.save_steps,
                experiment_dir=args.output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                # PEFT and quantization
                efficient_finetuning=args.efficient_finetuning,
                use_8bit_quantization=args.use_8bit_quantization,
            )
        case "DP-GReaT":
            synthesizer = GReaTDP(
                args.model,
                epochs=args.epochs,
                save_steps=args.save_steps,
                experiment_dir=args.output_dir,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                # Privacy
                per_sample_max_grad_norm=args.per_sample_max_grad_norm, 
                target_epsilon=args.target_epsilon, 
                target_delta=args.target_delta, 
                max_physical_batch_size=args.max_physical_batch_size,
                # PEFT and quantization
                efficient_finetuning=args.efficient_finetuning,
                use_8bit_quantization=args.use_8bit_quantization,
            )
        case "DP-LLMTGen":
            synthesizer = DPLLMTGen(
                args.model,
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                experiment_dir=args.output_dir,
                batch_size=args.batch_size,
                # DP-LLMTGen
                stage1_epochs = args.stage1_epochs,
                stage2_epochs = args.stage2_epochs,
                stage1_lr =args.stage1_lr,
                stage2_lr=args.stage2_lr,
                loss_alpha=args.loss_alpha,
                loss_beta=args.loss_beta,
                loss_lmbda=args.loss_lmbda,
                # Privacy
                per_sample_max_grad_norm=args.per_sample_max_grad_norm, 
                target_epsilon=args.target_epsilon, 
                target_delta=args.target_delta, 
                stage2_batch_size=args.max_physical_batch_size,
                # PEFT and quantization
                efficient_finetuning=args.efficient_finetuning,
                use_8bit_quantization=args.use_8bit_quantization,
            )
        case "LLMTGen":
            synthesizer = DPLLMTGen(
                args.model,
                save_steps=args.save_steps,
                logging_steps=args.logging_steps,
                experiment_dir=args.output_dir,
                batch_size=args.batch_size,
                # DP-LLMTGen
                stage1_epochs = args.stage1_epochs,
                stage2_epochs = args.stage2_epochs,
                stage1_lr =args.stage1_lr,
                stage2_lr=args.stage2_lr,
                loss_alpha=args.loss_alpha,
                loss_beta=args.loss_beta,
                loss_lmbda=args.loss_lmbda,
                use_dp=False,
                # Privacy
                per_sample_max_grad_norm=args.per_sample_max_grad_norm, 
                target_epsilon=args.target_epsilon, 
                target_delta=args.target_delta, 
                stage2_batch_size=args.max_physical_batch_size,
                # PEFT and quantization
                efficient_finetuning=args.efficient_finetuning,
                use_8bit_quantization=args.use_8bit_quantization,
            )
        case _:
            raise RuntimeError("Invalid method")

    # Train the model
    trainer = synthesizer.fit(df)

    # Save the model
    synthesizer.save(f"{output_dir}/final")

if __name__ == "__main__":
    args = parse_args()
    main(args)
