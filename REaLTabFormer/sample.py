import os
from pathlib import Path
from realtabformer import REaLTabFormer
import argparse
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description='Generate synthetic samples using REaLTabFormer models')
    parser.add_argument('--parent-model-path', type=str, required=True,
                       help='Path to the parent model directory')
    parser.add_argument('--child-model-path', type=str, required=True,
                       help='Path to the child model directory')
    parser.add_argument('--num-samples', type=int, required=True,
                       help='Number of samples to generate')
    parser.add_argument('--join-on', type=str, required=True,
                       help='Column name to join parent and child tables')
    parser.add_argument('--output-dir', type=str, default='synthesized',
                       help='Output directory for generated samples')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load both models
    parent_model = REaLTabFormer.load_from_dir(args.parent_model_path)
    child_model = REaLTabFormer.load_from_dir(args.child_model_path)

    # Generate parent samples
    parent_samples = parent_model.sample(args.num_samples)

    # Create the unique ids based on the index
    parent_samples.index.name = args.join_on
    parent_samples = parent_samples.reset_index()

    # Generate the relational observations
    child_samples = child_model.sample(
        input_unique_ids=parent_samples[args.join_on],
        input_df=parent_samples.drop(args.join_on, axis=1),
        gen_batch=36)

    # Create date for saving the samples
    date = datetime.now().strftime("%Y_%m_%d_%H_%M")

    # Save the samples
    parent_output = Path(args.output_dir) / f'parent_samples_{date}.csv'
    child_output = Path(args.output_dir) / f'child_samples_{date}.csv'
    
    parent_samples.to_csv(parent_output, index=False)
    child_samples.to_csv(child_output, index=False)

if __name__ == '__main__':
    main()
