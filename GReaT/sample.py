import logging
import os
from pathlib import Path
import argparse

import pandas as pd
from src import GReaT

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic samples using REaLTabFormer models')
    parser.add_argument('--model-path', type=str, required=True,
                       help='Path to the parent model directory')
    parser.add_argument('--output-dir', type=str, required=True, 
                        choices=["synthesized/GReaT", "synthesized/DP-GReaT", "synthesized/DP-LLMTGen", "synthesized/LLMTGen"],
                        help='Output directory for generated samples')
    parser.add_argument('--num-samples', type=int, default=1000,
                       help='Number of samples to generate')
    parser.add_argument('--train_df', type=str, default=None,
                        help="Path to the training dataset csv. Provide this only if you want the sample script to generate a dataset with the same size as the given dataset")
    parser.add_argument('--sample-batch-size', type=int, default=32,
                       help='Sampling batch size')
    parser.add_argument('--max-length', type=int, default=500,
                       help='Max length of the decode output.')
    parser.add_argument('--dataset-name', type=str, required=True,
                       help='Name of the dataset')
    parser.add_argument('--use_peft', action='store_true', help="whether the model uses PEFT")
    return parser.parse_args()

def main(args):
    
    output_path = Path(args.output_dir, args.dataset_name)
    os.makedirs(output_path, exist_ok=True)
    
    if args.use_peft:
        synthesizer = GReaT.load_peft_model(args.model_path)
    else:
        synthesizer = GReaT.load_from_dir(args.model_path)

    num_samples = args.num_samples
    if args.train_df:
        train_df = pd.read_csv(args.train_df)
        num_samples = len(train_df)

    samples = synthesizer.sample(num_samples, k=args.sample_batch_size, max_length=400)

    save_location = f"{args.output_dir}/{args.dataset_name}/sample_num_samples={num_samples}.csv"
    samples.to_csv(save_location)

    logging.info(f"Sample generated at {save_location}")
    

if __name__ == '__main__':
    args = parse_args()
    main(args)