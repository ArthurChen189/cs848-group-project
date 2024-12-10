import time
import pandas as pd
import os
from PostProcessor import RecordPostprocessor
import argparse

def generate_samples(trained_model, datasetType, num_samples, output_dir='synthesized/dp-stoa'):
    """Generate synthetic samples using the trained model."""
    
    print(f"\nGenerating {num_samples} synthetic samples...")
    sample_start_time = time.time()
    
    dl = trained_model['dataloader']
    gum = trained_model['gum']
    
    # Generate synthetic data
    syn_data = gum.synthesize(iterations=100, num_records=num_samples)
    
    # Post-processing the data
    processor_public = RecordPostprocessor(syn_data, dl.configpath, dl.datainfopath, dl.decode_mapping)
    synthesised_data = processor_public.post_process()
    
    # Sampling time completed
    sample_time = time.time() - sample_start_time
    print(f"Sampling completed in {sample_time:.2f} seconds")
    
    # Create directory if it doesn't exist
    if not os.path.exists(f'{output_dir}/{datasetType}'):
        os.makedirs(f'{output_dir}/{datasetType}')
    
    # Save the synthesized data
    output_path = f'{output_dir}/{datasetType}/samples_num_samples={num_samples}.csv'
    synthesised_data.to_csv(output_path, index=False)
    print(f"Generated {num_samples} synthetic samples and saved to {output_path}")
    
    return synthesised_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate synthetic samples using trained DP-STOA model')
    parser.add_argument('--dataset', type=str, default='adult',
                      choices=['adult', 'diabetes', 'california_housing'],
                      help='Dataset to generate samples for')
    parser.add_argument('--num_samples', type=int, default=0,
                      help='Number of samples to generate (0 for same as training data)')
    parser.add_argument('--data_dir', type=str, default='../data_config',
                      help='Directory containing dataset configurations')
    parser.add_argument('--output_dir', type=str, default='synthesized/dp-stoa',
                      help='Directory to save synthetic samples')
    
    args = parser.parse_args()
    
    # Train the model first
    from train import train_model
    trained_model = train_model(args.dataset, data_dir=args.data_dir)
    
    # If num_samples not specified, use same as training data
    if args.num_samples == 0:
        num_samples = len(pd.read_csv(f'{args.data_dir}/{args.dataset}/{args.dataset}_train.csv'))
    else:
        num_samples = args.num_samples
        
    synthetic_data = generate_samples(trained_model, args.dataset, num_samples, args.output_dir)