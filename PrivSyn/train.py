import time
from DataLoader import DataLoader
from Anonymisation import Anonymisation
from Consistenter import Consistenter
from GUM import GraduallyUpdateMethod
import os
import argparse

def train_model(datasetType, epsilon=0.5, delta=3e-11, data_dir='./data_config'):
    """Train the DP-STOA model and save the trained components."""
    
    print(f"\nTraining model for {datasetType} dataset...")
    train_start_time = time.time()
    
    # Initialize PrivSyn components
    dl = DataLoader(f'{data_dir}/{datasetType}/{datasetType}_train.csv', 
                   f'./data_config/{datasetType}/data.yaml', 
                   f'./data_config/{datasetType}/column_info.json', 
                   f'./data_config/{datasetType}/loading_data.json')
    dl.data_loader()
    dl.all_indifs(dl.private_data)
    
    # Anonymising the data
    anon = Anonymisation(epsilon=epsilon, delta=delta)
    anon.anonymiser(dl)
    
    # Consistenting the data
    cons = Consistenter(anon, dl.all_attrs)
    cons.make_consistent(iterations=5)
    
    # Initialising the Gradually Update Method
    gum = GraduallyUpdateMethod(dl, cons)
    gum.initialiser(view_iterations=100)
    
    # Training time completed
    train_time = time.time() - train_start_time
    print(f"Training completed in {train_time:.2f} seconds")
    
    # Create directory if it doesn't exist
    if not os.path.exists(f'models/{datasetType}'):
        os.makedirs(f'models/{datasetType}')
    
    return {
        'dataloader': dl,
        'gum': gum
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DP-STOA model')
    parser.add_argument('--dataset', type=str, default='adult',
                      choices=['adult', 'diabetes', 'california_housing'],
                      help='Dataset to train on')
    parser.add_argument('--epsilon', type=float, default=0.5,
                      help='Privacy parameter epsilon')
    parser.add_argument('--delta', type=float, default=3e-11,
                      help='Privacy parameter delta')
    parser.add_argument('--data_dir', type=str, default='./data_config',
                      help='Directory containing dataset configurations')
    
    args = parser.parse_args()
    trained_model = train_model(args.dataset, args.epsilon, args.delta, args.data_dir)