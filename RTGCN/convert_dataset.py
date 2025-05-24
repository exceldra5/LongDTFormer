import numpy as np
import pandas as pd
import pickle
import os
from scipy import sparse
from collections import defaultdict

def load_uci_data(base_path):
    """Load UCI dataset files"""
    # Load edge data
    edges_df = pd.read_csv(os.path.join(base_path, 'ml_CollegeMsg.csv'))
    # Rename columns to match our expected format
    edges_df = edges_df.rename(columns={
        'u': 'source',
        'i': 'target',
        'ts': 'timestamp'
    })
    
    # Load node data
    nodes = np.load(os.path.join(base_path, 'ml_CollegeMsg_node.npy'))
    
    # Load role data
    with open(os.path.join(base_path, 'merged_snapshot_factorized_roles.pkl'), 'rb') as f:
        roles_data = pickle.load(f)
    
    return edges_df, nodes, roles_data

def create_time_snapshots(edges_df, time_steps=10):
    """Create time snapshots from edge data"""
    # Sort by timestamp
    edges_df = edges_df.sort_values('timestamp')
    
    # Calculate time windows
    min_time = edges_df['timestamp'].min()
    max_time = edges_df['timestamp'].max()
    time_window = (max_time - min_time) / time_steps
    
    # Create snapshots
    snapshots = []
    for i in range(time_steps):
        start_time = min_time + i * time_window
        end_time = min_time + (i + 1) * time_window
        
        # Get edges in this time window
        snapshot_edges = edges_df[
            (edges_df['timestamp'] >= start_time) & 
            (edges_df['timestamp'] < end_time)
        ]
        snapshots.append(snapshot_edges)
    
    return snapshots

def create_adjacency_matrices(snapshots, num_nodes):
    """Create adjacency matrices for each snapshot"""
    adj_matrices = []
    
    for snapshot in snapshots:
        # Create sparse adjacency matrix
        rows = snapshot['source'].values
        cols = snapshot['target'].values
        data = np.ones(len(rows))
        
        adj_matrix = sparse.coo_matrix(
            (data, (rows, cols)),
            shape=(num_nodes, num_nodes)
        ).toarray()
        
        # Make it symmetric (undirected)
        adj_matrix = np.maximum(adj_matrix, adj_matrix.T)
        adj_matrices.append(adj_matrix)
    
    return np.array(adj_matrices)

def create_attribute_matrices(nodes, time_steps, feature_dim=100):
    """Create attribute matrices for each snapshot"""
    num_nodes = len(nodes)
    # Create random features for each node with 100 dimensions
    node_features = np.random.randn(num_nodes, feature_dim)
    # Normalize features
    node_features = node_features / np.linalg.norm(node_features, axis=1, keepdims=True)
    # Repeat for each time step
    attmats = np.stack([node_features] * time_steps, axis=1)
    return attmats

def create_role_dictionary(roles_data, time_steps):
    """Create role dictionary in DBLP format"""
    role_dict = {}
    
    # Convert roles_data to DBLP format
    # roles_data is a dict with keys 1-29 (role IDs)
    # Each value contains node assignments for that role
    
    # Create time steps
    for t in range(time_steps):
        role_dict[t] = {}
        # For each role in the original data
        for role_id, nodes in roles_data.items():
            # Convert role_id to string to match DBLP format
            role_dict[t][str(role_id)] = nodes.tolist() if isinstance(nodes, np.ndarray) else nodes
    
    return role_dict

def convert_uci_to_dblp_format(uci_path, output_path, time_steps=10):
    """Convert UCI dataset to DBLP format"""
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load UCI data
    edges_df, nodes, roles_data = load_uci_data(uci_path)
    
    # Get number of nodes
    num_nodes = len(nodes)
    
    # Create time snapshots
    snapshots = create_time_snapshots(edges_df, time_steps)
    
    # Create adjacency matrices
    adjs = create_adjacency_matrices(snapshots, num_nodes)
    
    # Create attribute matrices
    attmats = create_attribute_matrices(nodes, time_steps)
    
    # Create role dictionary
    role_dict = create_role_dictionary(roles_data, time_steps)
    
    # Create simple binary labels
    labels = np.zeros(num_nodes)
    
    # Save DBLP format files
    # Save .npz file
    np.savez(
        os.path.join(output_path, 'UCI.npz'),
        adjs=adjs,
        attmats=attmats,
        labels=labels
    )
    
    # Save role dictionary
    with open(os.path.join(output_path, 'UCI_wl_nc.pkl'), 'wb') as f:
        pickle.dump(role_dict, f)
    
    print(f"Conversion complete. Files saved in {output_path}")
    print(f"Number of nodes: {num_nodes}")
    print(f"Number of time steps: {time_steps}")
    print(f"Adjacency matrices shape: {adjs.shape}")
    print(f"Attribute matrices shape: {attmats.shape}")
    print(f"Labels shape: {labels.shape}")

if __name__ == "__main__":
    # Paths
    uci_path = "./data/UCI"
    output_path = "./data/UCI"
    
    # Convert dataset with 12 time steps
    convert_uci_to_dblp_format(uci_path, output_path, time_steps=12) 