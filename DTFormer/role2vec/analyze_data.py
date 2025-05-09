import os
import sys
import pickle
import argparse

import numpy as np
import pandas as pd
from tabulate import tabulate


def analyze_roles(dataset: str = 'CollegeMsg'):
    # dataset = 'CollegeMsg'
    
    grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(grandparent_dir, 'output', dataset)
    data_file = os.path.join(data_dir, 'merged_snapshot_factorized_roles.pkl')
    
    if not os.path.exists(data_file):
        print(f"File {data_file} does not exist.")
        return
    
    with open(data_file, "rb") as f:
        graph = pickle.load(f)
    
    for time_step in sorted(graph.keys()):
        print(f"\nTime Step: {time_step}")
        
        node_roles = graph[time_step]
        
        # Print first 5 nodes and their roles
        for i, (node_id, role) in enumerate(node_roles.items()):
            if i >= 5:
                break
            print(f"  Node ID: {node_id}, Role: {role}")
                
            
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Analyze roles in the dataset.")
    argparser.add_argument("--dataset", type=str, default="CollegeMsg", help="Dataset name")
    args = argparser.parse_args()
    
    analyze_roles(dataset=args.dataset)