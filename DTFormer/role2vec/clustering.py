import os
import pickle
import re
import numpy as np
import pandas as pd
import networkx as nx
import argparse # Needed to create the args object
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
import math # Needed if any part of motif counting is implicitly called
import random


# --- Helper: Mock MotifCounterMachine to call factorize_string_matrix ---
class MockMotifMachine:
    """
    A mock class to hold necessary attributes (graph, binned_features, args)
    and execute the factorize_string_matrix method.
    """
    def __init__(self, graph, binned_features, args):
        self.graph = graph
        self.binned_features = binned_features
        self.args = args

    # --- The factorize_string_matrix method copied from your prompt ---
    def factorize_string_matrix(self):
        """
        Creating string labels by factorization.
        Handles mapping between original node IDs and contiguous indices
        for sparse matrix and clustering.
        """
        # Get the list of nodes in a consistent order for indexing
        # Convert nodes to a list and get original node IDs
        node_list = list(self.graph.nodes())
        num_nodes = len(node_list)
        # Create a mapping from original node ID to its contiguous index (0 to num_nodes-1)
        node_to_internal_index = {node: i for i, node in enumerate(node_list)}

        rows = [] # Will store internal indices (0 to num_nodes-1)
        columns = [] # Will store feature indices (based on quantile bin + motif index)
        scores = [] # Will store the value (always 1 for presence)

        # Iterate through features using the ordered node list
        # self.binned_features stores features keyed by *string* node IDs
        # We need to check if the node is in self.binned_features as some nodes
        # might exist in the graph but have no motif features (e.g., isolated)
        for node in node_list:
            node_str = str(node)
            if node_str in self.binned_features:
                 features_for_node = self.binned_features[node_str]
                 node_internal_index = node_to_internal_index[node]

                 for feature_str in features_for_node:
                     # Ensure the feature string can be converted to an integer index
                     try:
                         feature_index = int(feature_str)
                         rows.append(node_internal_index)
                         columns.append(feature_index)
                         scores.append(1)
                     except ValueError:
                         # This should ideally not happen if binned_features are correctly structured
                         print(f"Warning: Could not convert feature string '{feature_str}' to int for node {node}. Skipping feature.")
                         continue


        # Handle cases where no features or nodes are present
        if num_nodes == 0 or not rows:
            print("Warning: No nodes in graph or no motif features generated for any node. Returning default role 'no_role'.")
            # Return default role for all nodes in the graph, if any
            return {str(node): "no_role" for node in self.graph.nodes()}


        row_number = num_nodes # The number of nodes is the size of the first dimension
        column_number = max(columns)+1 if columns else 0 # Max feature index + 1

        # Handle case with no columns (e.g., no unique features generated)
        if column_number == 0:
            print("Warning: No unique motif feature indices found. Assigning default role 'no_features'.")
            return {str(node): "no_features" for node in self.graph.nodes()}


        features_matrix = csr_matrix((scores, (rows, columns)), shape=(row_number, column_number))

        # Handle case where NMF n_components > features.shape[1]
        # NMF requires n_components <= n_features (column_number)
        n_components = min(self.args.factors, column_number)
        if n_components <= 0:
            print(f"Warning: Effective number of factors ({n_components}) is zero or negative (requested {self.args.factors}, max_features {column_number}). Assigning default role 'no_factors'.")
            return {str(node): "no_factors" for node in self.graph.nodes()}

        # Handle case where clusters > number of samples (rows = num_nodes)
        n_clusters = min(self.args.clusters, num_nodes)
        if n_clusters <= 1: # Need at least 2 clusters for meaningful clustering
            print(f"Warning: Effective number of clusters ({n_clusters}) is less than 2 (requested {self.args.clusters}, num_nodes {num_nodes}). Assigning a single default role 'single_cluster'.")
            # Assign a single role if only one cluster is possible
            return {str(node): "single_cluster" for node in self.graph.nodes()}


        try:
            # Use the adjusted number of components
            model = NMF(n_components=n_components, init="random", random_state=self.args.seed, alpha_H=self.args.beta, alpha_W=self.args.beta, max_iter=10000)
            factors = model.fit_transform(features_matrix)

            # Use the adjusted number of clusters
            # Added n_init for robustness in KMeans initialization
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed, n_init=10).fit(factors)
            labels = kmeans.labels_ # This array is indexed 0 to num_nodes-1

        except Exception as e:
            print(f"Error during NMF/KMeans for snapshot: {e}. Assigning default role 'clustering_failed'.")
            # Assign a default role if clustering fails
            return {str(node): "clustering_failed" for node in self.graph.nodes()}


        # Map the cluster labels (indexed 0 to num_nodes-1) back to original node IDs
        final_features = {}
        for i, node in enumerate(node_list):
            final_features[str(node)] = str(labels[i]) # labels[i] corresponds to node_list[i]

        return final_features


# --- Reconstruct binned_features and call factorization for a snapshot ---
def process_snapshot_for_factorization(snapshot_id: int, graph_df: pd.DataFrame, motif_fact_args: argparse.Namespace):
    """
    Loads inverted string roles for a snapshot, reconstructs binned_features,
    builds the graph, and runs factorization.

    :param snapshot_id: The ID of the snapshot (integer).
    :param graph_df: The full DataFrame with all edges and 'snapshots' column.
    :param motif_fact_args: Arguments for motif factorization.
    :return: A dictionary {node_id: factorized_role_label_str} for this snapshot.
             Returns an empty dictionary if processing fails or no data.
    """
    print(f"Processing snapshot {snapshot_id} for factorization...")

    # 1. Build the cumulative graph for this snapshot
    snapshot_edges_df = graph_df[graph_df['snapshots'] <= snapshot_id]

    if snapshot_edges_df.empty:
        print(f"No edges found up to snapshot {snapshot_id}. Skipping.")
        return {}

    snapshot_graph = nx.from_edgelist(snapshot_edges_df[['u', 'i']].values.tolist())
    snapshot_graph.remove_edges_from(nx.selfloop_edges(snapshot_graph))

    if snapshot_graph.number_of_nodes() == 0:
         print(f"No nodes in graph for snapshot {snapshot_id}. Skipping.")
         return {}

    # 2. Load the inverted string roles for this snapshot
    # Construct the expected filename (e.g., snapshot_0001.pkl)
    filename = f'snapshot_{snapshot_id:04d}.pkl' # Assuming 4 digits padding
    file_path = os.path.join(temp_motif_roles_dir, filename)

    if not os.path.exists(file_path):
        print(f"Inverted roles file not found for snapshot {snapshot_id}: {file_path}. Exiting.")
        exit()

    try:
        with open(file_path, 'rb') as f:
            inverted_roles_this_snapshot = pickle.load(f)
    except Exception as e:
        print(f"Error loading inverted roles from {file_path}: {e}. Exiting.")
        exit()

    # 3. Reconstruct self.binned_features
    # The inverted roles map concatenated_string -> [list_of_nodes]
    # We need str(node) -> [list_of_individual_feature_strings]
    reconstructed_binned_features = {}
    for concatenated_role_str, node_ids_list in inverted_roles_this_snapshot.items():
        # Split the concatenated string back into individual feature strings
        # Handle cases where the string might be empty or not contain '_'
        individual_feature_strings = concatenated_role_str.split('_') if '_' in concatenated_role_str else [concatenated_role_str]

        for node_id in node_ids_list:
            # Convert node_id to string key as expected by binned_features
            reconstructed_binned_features[str(node_id)] = individual_feature_strings

    # 4. Instantiate mock machine and run factorization
    mock_machine = MockMotifMachine(snapshot_graph, reconstructed_binned_features, motif_fact_args)

    try:
        factorization_roles_raw = mock_machine.factorize_string_matrix()
    except Exception as e:
        print(f"Error during factorization for snapshot {snapshot_id}: {e}.")
        return {}

    # 5. Format the output: convert string keys back to original node type (longlong)
    factorization_roles = {}
    for node_str, role_label_str in factorization_roles_raw.items():
        # Convert node ID back to int/longlong
        node_id = np.longlong(node_str)
        factorization_roles[node_id] = role_label_str

    return factorization_roles


class Data:
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, snapshots: np.ndarray):
        """
        Data object to store the nodes interaction information.
        :param src_node_ids: ndarray
        :param dst_node_ids: ndarray
        :param node_interact_times: ndarray
        :param edge_ids: ndarray
        :param labels: ndarray
        """
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.snapshots = snapshots


def get_link_prediction_data(dataset_name: str, val_ratio: float, test_ratio: float, num_snapshots: int):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
            node_snap_counts (np.ndarray) - Added as per your original snippet
            graph_df (pd.DataFrame) - Added for motif role processing
    """
    # Load data and train val test split
    grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_df_path = os.path.join(grandparent_dir, 'processed_data', dataset_name, f'ml_{dataset_name}.csv')
    edge_feat_path = os.path.join(grandparent_dir, 'processed_data', dataset_name, f'ml_{dataset_name}.npy')
    node_feat_path = os.path.join(grandparent_dir, 'processed_data', dataset_name, f'ml_{dataset_name}_node.npy')

    try:
        graph_df = pd.read_csv(graph_df_path)
        edge_raw_features = np.load(edge_feat_path)
        node_raw_features = np.load(node_feat_path)
        print(f"Successfully loaded data for dataset: {dataset_name}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Attempted paths: {graph_df_path}, {edge_feat_path}, {node_feat_path}")
        print("Returning dummy data for demonstration. Please ensure dataset files exist for real use.")
        exit(1)

    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    # Padding features to a fixed dimension (172)
    # Need to handle potential dimension mismatches gracefully if using dummy data
    current_node_feat_dim = node_raw_features.shape[1]
    current_edge_feat_dim = edge_raw_features.shape[1]

    if current_node_feat_dim < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - current_node_feat_dim))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    # If node features are already larger, truncate or handle appropriately. Current padding logic assumes smaller.
    # For this use case, we'll assume padding is always needed for smaller dimensions based on the original code.
    # If current_node_feat_dim > NODE_FEAT_DIM: print("Warning: Node feature dim > target, not padding.")

    if current_edge_feat_dim < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - current_edge_feat_dim))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)
    # If current_edge_feat_dim > EDGE_FEAT_DIM: print("Warning: Edge feature dim > target, not padding.")


    # get the timestamp of validate and test set
    min_ts = graph_df['ts'].min() if not graph_df.empty else 0
    max_ts = graph_df['ts'].max() if not graph_df.empty else 1 # Avoid division by zero

    # Ensure graph_df has enough data points for quantiles or use min/max if not
    if len(graph_df) < 2: # Need at least 2 points for quantile range
         val_time = min_ts + (max_ts - min_ts) * (1 - val_ratio - test_ratio)
         test_time = min_ts + (max_ts - min_ts) * (1 - test_ratio)
    else:
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))


    # Calculate snapshot indices
    if max_ts == min_ts:
        # If all timestamps are the same, assign all to snapshot 1
        graph_df['snapshots'] = 1
        range_size = 1 # Avoid division by zero
    else:
        range_size = (max_ts - min_ts) / num_snapshots
        # Assign snapshots starting from 1
        graph_df['snapshots'] = ((graph_df['ts'] - min_ts) / range_size).astype(np.int16) + 1

    # Cap snapshots at num_snapshots and ensure minimum is 1
    graph_df.loc[graph_df['snapshots'] > num_snapshots, 'snapshots'] = num_snapshots
    graph_df.loc[graph_df['snapshots'] < 1, 'snapshots'] = 1


    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    snapshots = graph_df.snapshots.values

    # Prepare data for node_snap_counts
    if not graph_df.empty:
        df_long = pd.concat([
            graph_df.rename(columns={'u': 'node'})[['node', 'snapshots']],
            graph_df.rename(columns={'i': 'node'})[['node', 'snapshots']]
        ])
        node_counts_per_snapshot = df_long.groupby(['node', 'snapshots']).size().unstack(fill_value=0)
        all_nodes = np.sort(np.unique(graph_df[['u', 'i']].values))
    else:
        node_counts_per_snapshot = pd.DataFrame()
        all_nodes = np.array([])

    all_snapshots_range = np.arange(1, num_snapshots + 1)

    # Ensure index and columns cover all possible nodes and snapshots
    node_counts_per_snapshot = node_counts_per_snapshot.reindex(index=all_nodes, columns=all_snapshots_range, fill_value=0)

    node_snap_counts = node_counts_per_snapshot.values

    # Add a zero vector row for node 0 if needed, or just handle empty case
    if node_snap_counts.shape[0] == 0:
        # If no nodes exist, create an empty array with the correct number of snapshot columns
        zero_vector = np.zeros((1, all_snapshots_range.shape[0]))
    else:
        # If nodes exist, add a zero row for node 0 if node IDs start from 0 and 0 is not present, or if needed structurally
        # For simplicity mirroring the original vstack:
        zero_vector = np.zeros((1, node_snap_counts.shape[1]))

    node_snap_counts = np.vstack([zero_vector, node_snap_counts])


    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, snapshots=snapshots)

    # the setting of seed follows previous works
    random.seed(2020)

    train_mask = (node_interact_times <= val_time)
    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], snapshots=snapshots[train_mask])

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask], snapshots=snapshots[val_mask])

    test_mask = node_interact_times > test_time
    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask], snapshots=snapshots[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))

    # Pass graph_df directly as it contains snapshots info needed for iteration
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, None, None, node_snap_counts, graph_df


# --- Main Execution ---
if __name__ == "__main__":
    # --- Configuration ---
    argparse = argparse.ArgumentParser(description="Factorize motif roles for each snapshot")
    argparse.add_argument('--dataset', type=str, default='CollegeMsg', help='Dataset name')
    args = argparse.parse_args()

    grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_motif_roles_dir = os.path.join(grandparent_dir, 'output', args.dataset, 'temp_motif_roles')
    output_filename = 'merged_snapshot_factorized_roles.pkl' # New output file for factorization results

    # Motif factorization parameters (should match what you want to use now)
    # These are defaults from param_parser.py, adjust if needed
    MOTIF_FACT_ARGS = {
        "graphlet_size": 4,
        "quantiles": 5,
        "motif_compression": "factorization",
        "factors": 8,
        "clusters": 50,
        "beta": 0.01,
        "seed": 42
    }
    vars(args).update(MOTIF_FACT_ARGS)

    # --- Script Logic Starts Here ---
    # Define dataset parameters to get graph_df and num_snapshots
    val_ratio = 0.1
    test_ratio = 0.1
    num_snapshots = 29 # Define the number of snapshots you processed originally

    # Path Calculation
    temp_dir_path = os.path.abspath(temp_motif_roles_dir)
    parent_dir_path = os.path.dirname(temp_dir_path)
    output_file_path = os.path.join(parent_dir_path, output_filename)

    # Step 1: Load essential data (graph_df) using your loader
    # We only need graph_df for building snapshot graphs and num_snapshots
    print("Loading data to get graph structure...")
    try:
        # Assuming your get_link_prediction_data returns graph_df as the last element
        _, _, _, _, _, _, _, _, _, graph_df = \
            get_link_prediction_data(args.dataset, val_ratio, test_ratio, num_snapshots)
        print("Data loading complete.")

        # Ensure graph_df contains the necessary columns
        if not all(col in graph_df.columns for col in ['u', 'i', 'ts', 'snapshots']):
             raise ValueError("Loaded graph_df is missing required columns ('u', 'i', 'ts', 'snapshots').")

        # Verify num_snapshots from data matches config (optional)
        # actual_max_snapshot = graph_df['snapshots'].max() if not graph_df.empty else 0
        # if actual_max_snapshot > num_snapshots:
        #     print(f"Warning: Data contains snapshots up to {actual_max_snapshot}, but config is {num_snapshots}.")
        #     print("Using config num_snapshots to iterate through existing files.")

    except Exception as e:
        print(f"Error loading graph data: {e}")
        print("Cannot proceed without graph structure information. Exiting.")
        exit()


    # Step 2: Process each snapshot file and run factorization
    all_snapshots_factorized_roles = {}

    print(f"\nStarting factorization for {num_snapshots} snapshots...")

    for snapshot_id in range(1, num_snapshots + 1):
         # The function process_snapshot_for_factorization handles loading the pkl
         # and performing the factorization for this specific snapshot.
         snapshot_factorized_roles = process_snapshot_for_factorization(
             snapshot_id,
             graph_df, # Pass the full graph_df to build the snapshot graph
             args # Pass the factorization arguments
         )
         all_snapshots_factorized_roles[snapshot_id] = snapshot_factorized_roles


    # Step 3: Save the final merged factorized roles
    print(f"\nFinished processing all snapshots. Saving merged factorized data to: {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(all_snapshots_factorized_roles, f)
        print("Merged factorized data saved successfully.")

    except Exception as e:
        print(f"Error saving merged factorized data: {e}")