import sys
import os
import numpy as np
import pandas as pd
from tabulate import tabulate
import io # To capture print output from the data loader

# Import the data loading function
# This assumes the script is run from the repository root directory
try:
    from utils.DataLoader import get_link_prediction_data
except ImportError:
    print("Error: Could not import get_link_prediction_data.")
    print("Please ensure you are running the script from the repository root")
    print("or that the 'utils' directory is in your Python path.")
    sys.exit(1)

# Define the datasets to analyze and their corresponding snapshot counts
# from the data_snapshots_num dictionary in train_link_prediction.py
datasets_to_analyze = [
    'bitcoinalpha',
    'bitcoinotc',
    'CollegeMsg',
    'reddit-body',
    'reddit-title',
    'mathoverflow',
    'email-Eu-core',
    # Add more datasets as needed
    'CanParl',
    'Contacts',
    'enron',
    'Flights',
    'Iastfm',
    'mooc',
    'myket',
    'SocialEvo',
    'uci',
    'UNtrade',
    'UNvote',
    'USLegis',
    'wikipedia'
]

# Snapshot counts as defined in the original train_link_prediction.py
data_snapshots_num = {
    'bitcoinalpha': 274,
    'bitcoinotc': 279,
    'CollegeMsg': 29,
    'reddit-body': 178,
    'reddit-title': 178,
    'mathoverflow': 2350,
    'email-Eu-core': 803,
    # Add more datasets as needed
    'CanParl': 100,
    'Contacts': 100,
    'enron': 100,
    'Flights': 100,
    'Iastfm': 100,
    'mooc': 100,
    'myket': 100,
    'SocialEvo': 100,
    'uci': 100,
    'UNtrade': 100,
    'UNvote': 100,
    'USLegis': 100,
    'wikipedia:': 100
}

# Fixed parameters for the get_link_prediction_data function
# These don't affect the overall statistics of the full dataset
val_ratio = 0.1
test_ratio = 0.1

results = []

print("Analyzing datasets...")

for dataset_name in datasets_to_analyze:
    print(f"Processing {dataset_name}...")
    try:
        # get_link_prediction_data prints information during loading.
        # We redirect stdout temporarily to avoid cluttering the output.
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Load the full data using the existing data loading function
        # We only need the full_data object for its statistics
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _, node_snap_counts = \
            get_link_prediction_data(dataset_name=dataset_name, val_ratio=val_ratio, test_ratio=test_ratio,
                                     num_snapshots=data_snapshots_num.get(dataset_name, None)) # Use .get() in case a dataset is missing from the map

        # Restore stdout
        sys.stdout = old_stdout

        # Extract or calculate the desired statistics from the full_data object
        num_nodes = full_data.num_unique_nodes
        num_interactions = full_data.num_interactions
        num_timestamps = data_snapshots_num.get(dataset_name, "N/A") # Get from the predefined map
        
        # Calculate average degree: (total number of incident edges) / (number of nodes)
        # In an undirected graph of V nodes and E edges, sum of degrees is 2*E.
        # Here, 'Interactions' represents the total number of events. Assuming each event
        # involves two nodes, the total "degree-sum" contributed by interactions is 2 * num_interactions.
        # Average degree is (2 * num_interactions) / num_nodes.
        avg_degree = (2 * num_interactions) / num_nodes if num_nodes > 0 else 0

        results.append([
            dataset_name,
            f"{num_nodes:,}", # Format with commas
            f"{num_interactions:,}", # Format with commas
            num_timestamps,
            f"{avg_degree:.2f}" # Format average degree to 2 decimal places
        ])

    except FileNotFoundError:
        # Restore stdout before printing error messages
        sys.stdout = old_stdout
        print(f"Error: Data files not found for dataset '{dataset_name}'.")
        print(f"Please ensure the './processed_data/{dataset_name}/' directory exists")
        print("and contains the necessary files (.csv and .npy). Skipping.")
        results.append([dataset_name, "N/A", "N/A", "N/A", "N/A"])
    except KeyError:
        # Restore stdout before printing error messages
        sys.stdout = old_stdout
        print(f"Error: Snapshot count not found for dataset '{dataset_name}'. Skipping.")
        results.append([dataset_name, "N/A", "N/A", "N/A", "N/A"])
    except Exception as e:
        # Restore stdout before printing error messages
        sys.stdout = old_stdout
        print(f"An unexpected error occurred while processing dataset '{dataset_name}': {e}. Skipping.")
        results.append([dataset_name, "Error", "Error", "Error", "Error"])

# Print the results in a formatted table
print("\n" + "="*40)
print("Dataset Statistics:")
print("="*40)
headers = ["Datasets", "Nodes", "Interactions", "Timestamps", "Average Degree"]
# Use 'grid' table format for better readability
print(tabulate(results, headers=headers, tablefmt="grid"))

# Optional: You might want to add 'Edges' if it's defined differently
# in your context (e.g., unique node pairs with at least one interaction).
# However, the current code doesn't compute this easily from the full_data object.