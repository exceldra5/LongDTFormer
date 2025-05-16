import os
import pickle
import re
import numpy as np
import pandas as pd
import networkx as nx
import argparse
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm import tqdm
import math
import random
import warnings

warnings.filterwarnings("ignore", message="Number of distinct clusters is less than 2. "
                                          "Results may appear meaningless.")


# --- Helper: Find Optimal Clusters using Silhouette Score ---
def find_optimal_clusters(data_matrix: np.ndarray, min_clusters: int, max_clusters: int, random_state: int, n_init: int = 10):
    """
    Finds the optimal number of clusters for a data matrix within a range
    using the silhouette score.

    :param data_matrix: The data matrix (NMF factors) to cluster.
    :param min_clusters: Minimum number of clusters to test (must be >= 2).
    :param max_clusters: Maximum number of clusters to test.
    :param random_state: Random state for KMeans.
    :param n_init: Number of initializations for KMeans.
    :return: The optimal number of clusters, or min_clusters if no valid options >= 2 exist.
    """
    num_samples = data_matrix.shape[0]

    if num_samples < 2:
        print("Warning: Not enough samples for clustering (less than 2). Returning min_clusters (will likely be ignored).")
        return min_clusters

    search_max = min(max_clusters, num_samples -1)
    search_min = max(min_clusters, 2)

    if search_max < search_min:
         print(f"Warning: Search range for clusters is invalid [{search_min}, {search_max}]. Cannot perform silhouette analysis. Returning {search_min}.")
         return search_min

    silhouette_scores = {}

    print(f"Searching for optimal clusters in range [{search_min}, {search_max}] using silhouette score...")

    for n_clusters in tqdm(range(search_min, search_max + 1), desc="Finding best n_clusters"):
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
            kmeans.fit(data_matrix)
            score = silhouette_score(data_matrix, kmeans.labels_)
            silhouette_scores[n_clusters] = score
        except Exception:
            pass

    if not silhouette_scores:
        print(f"Warning: No valid silhouette scores computed in range [{search_min}, {search_max}]. Returning default {search_min}.")
        return search_min

    optimal_k = max(silhouette_scores, key=silhouette_scores.get)
    print(f"Found optimal clusters: {optimal_k} (Silhouette Score: {silhouette_scores[optimal_k]:.4f})")
    return optimal_k


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

    def factorize_string_matrix(self):
        """
        Creating string labels by factorization.
        Handles mapping between original node IDs and contiguous indices
        for sparse matrix and clustering.
        Uses either node_based or auto (silhouette) method for n_clusters.
        """
        node_list = list(self.graph.nodes())
        num_nodes = len(node_list)

        if num_nodes == 0:
             print("Warning: No nodes in graph. Returning default role 'no_role'.")
             return {}

        node_to_internal_index = {node: i for i, node in enumerate(node_list)}

        rows = []
        columns = []
        scores = []

        processed_nodes_count = 0
        for node in node_list:
            node_str = str(node)
            if node_str in self.binned_features and self.binned_features[node_str]:
                 features_for_node = self.binned_features[node_str]
                 node_internal_index = node_to_internal_index[node]

                 processed_nodes_count += 1
                 for feature_str in features_for_node:
                     try:
                         feature_index = int(feature_str)
                         rows.append(node_internal_index)
                         columns.append(feature_index)
                         scores.append(1)
                     except ValueError:
                         print(f"Warning: Could not convert feature string '{feature_str}' to int for node {node}. Skipping feature.")
                         continue


        if not rows or processed_nodes_count < 2:
            print(f"Warning: Only {processed_nodes_count} nodes with motif features generated in this snapshot (need >= 2 for clustering). Returning default role 'no_features' for these nodes.")
            return {str(node): ("no_features" if str(node) in self.binned_features and self.binned_features[str(node)] else "no_role") for node in self.graph.nodes()}

        row_number = num_nodes
        column_number = max(columns)+1 if columns else 0

        if column_number == 0:
            print("Warning: No unique motif feature indices found. Assigning default role 'no_features'.")
            return {str(node): "no_features" for node in self.graph.nodes()}

        features_matrix = csr_matrix((scores, (rows, columns)), shape=(row_number, column_number))

        effective_n_factors = min(self.args.factors, column_number, features_matrix.shape[0])
        if effective_n_factors <= 0:
            print(f"Warning: Effective number of factors ({effective_n_factors}) is zero or negative. Assigning default role 'no_factors'.")
            return {str(node): "no_factors" for node in self.graph.nodes()}

        try:
            model = NMF(n_components=effective_n_factors, init="random", random_state=self.args.seed, alpha_H=self.args.beta, alpha_W=self.args.beta, max_iter=10000)
            factors = model.fit_transform(features_matrix)
        except Exception as e:
            print(f"Error during NMF for snapshot: {e}. Assigning default role 'nmf_failed'.")
            return {str(node): "nmf_failed" for node in self.graph.nodes()}

        n_clusters = 0
        min_clusters_search = 2

        if self.args.cluster_method == 'node_based':
            n_clusters = max(min_clusters_search, len(self.graph.nodes()) // 100)
            n_clusters = min(n_clusters, factors.shape[0])
            print(f"Using node-based clustering: {n_clusters} clusters requested.")

        elif self.args.cluster_method == 'auto':
            max_clusters_search = min(self.args.clusters_max_search, factors.shape[0] - 1, factors.shape[1] if factors.shape[1] > 0 else 0)
            max_clusters_search = max(max_clusters_search, min_clusters_search)

            n_clusters = find_optimal_clusters(
                factors,
                min_clusters=min_clusters_search,
                max_clusters=max_clusters_search,
                random_state=self.args.seed,
                n_init=10
            )
            # if factors.shape[0] >= 2:
            #      n_clusters = max(n_clusters, 2)
            # else:
            #      n_clusters = 1
        elif self.args.cluster_method == 'max':
            max_clusters_search = self.args.clusters_max_search
            
            n_clusters = find_optimal_clusters(
                factors,
                min_clusters=min_clusters_search,
                max_clusters=max_clusters_search,
                random_state=self.args.seed,
                n_init=10
            )
            # if factors.shape[0] >= 2:
            #      n_clusters = max(n_clusters, 2)
            # else:
            #      n_clusters = 1
        else:
            raise ValueError(f"Unknown cluster_method: {self.args.cluster_method}. Must be 'node_based' or 'auto'.")

        if n_clusters < 2 or n_clusters > factors.shape[0]:
             print(f"Warning: Final determined number of clusters ({n_clusters}) is invalid for {factors.shape[0]} samples. Assigning a single default role 'invalid_cluster_num'.")
             return {str(node): "invalid_cluster_num" for node in self.graph.nodes()}

        try:
            kmeans = KMeans(n_clusters=int(n_clusters), random_state=self.args.seed, n_init=10).fit(factors)
            labels = kmeans.labels_

        except Exception as e:
            print(f"Error during KMeans for snapshot: {e}. Assigning default role 'clustering_failed'.")
            return {str(node): "clustering_failed" for node in self.graph.nodes()}

        features_matrix_row_to_original_node = {}
        current_matrix_row_index = 0
        for i_graph, node in enumerate(node_list):
             node_str = str(node)
             if node_str in self.binned_features and self.binned_features[node_str]:
                 features_matrix_row_to_original_node[current_matrix_row_index] = node
                 current_matrix_row_index += 1

        final_features = {}
        for matrix_row_index, original_node_id in features_matrix_row_to_original_node.items():
             final_features[str(original_node_id)] = str(labels[matrix_row_index])

        nodes_with_features = set(features_matrix_row_to_original_node.values())
        for node in node_list:
            if node not in nodes_with_features:
                 final_features[str(node)] = "no_features"


        return final_features, n_clusters


# --- Data Loading and Snapshot Processing ---

class Data:
    def __init__(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray, edge_ids: np.ndarray, labels: np.ndarray, snapshots: np.ndarray):
        """
        Data object to store the nodes interaction information.
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
    """
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
        print("Please ensure dataset files exist for real use.")
        empty_df = pd.DataFrame(columns=['u', 'i', 'ts', 'idx', 'label'])
        empty_node_feat = np.zeros((1, 172))
        empty_edge_feat = np.zeros((1, 172))
        print("Returning empty dataframes/arrays.")
        return empty_node_feat, empty_edge_feat, None, None, None, None, None, None, np.zeros((1, num_snapshots)), empty_df


    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    current_node_feat_dim = node_raw_features.shape[1]
    current_edge_feat_dim = edge_raw_features.shape[1]

    if current_node_feat_dim < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - current_node_feat_dim))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)

    if current_edge_feat_dim < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - current_edge_feat_dim))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)


    min_ts = graph_df['ts'].min() if not graph_df.empty else 0
    max_ts = graph_df['ts'].max() if not graph_df.empty else 1

    if not graph_df.empty and len(graph_df) >= 2:
        val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))
    else:
        val_time = min_ts + (max_ts - min_ts) * (1 - val_ratio - test_ratio) if max_ts > min_ts else min_ts
        test_time = min_ts + (max_ts - min_ts) * (1 - test_ratio) if max_ts > min_ts else min_ts


    if max_ts == min_ts:
        graph_df['snapshots'] = 1 if not graph_df.empty else 0
        range_size = 1
    else:
        range_size = (max_ts - min_ts) / num_snapshots
        graph_df['snapshots'] = ((graph_df['ts'] - min_ts) / range_size).astype(np.int16) + 1

    if not graph_df.empty:
        graph_df.loc[graph_df['snapshots'] > num_snapshots, 'snapshots'] = num_snapshots
        graph_df.loc[graph_df['snapshots'] < 1, 'snapshots'] = 1
    else:
         graph_df['snapshots'] = 0


    src_node_ids = graph_df.u.values.astype(np.longlong) if not graph_df.empty else np.array([], dtype=np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong) if not graph_df.empty else np.array([], dtype=np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64) if not graph_df.empty else np.array([], dtype=np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong) if not graph_df.empty else np.array([], dtype=np.longlong)
    labels = graph_df.label.values if not graph_df.empty else np.array([], dtype=int)
    snapshots = graph_df.snapshots.values if not graph_df.empty else np.array([], dtype=int)

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

    node_counts_per_snapshot = node_counts_per_snapshot.reindex(index=all_nodes, columns=all_snapshots_range, fill_value=0)

    node_snap_counts_values = node_counts_per_snapshot.values

    zero_vector = np.zeros((1, all_snapshots_range.shape[0]))
    node_snap_counts = np.vstack([zero_vector, node_snap_counts_values])


    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, snapshots=snapshots)

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

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, None, None, node_snap_counts, graph_df


# --- Reconstruct binned_features and call factorization for a snapshot ---
def process_snapshot_for_factorization(snapshot_id: int, graph_df: pd.DataFrame, motif_fact_args: argparse.Namespace, temp_motif_roles_dir: str):
    """
    Loads inverted string roles for a snapshot, reconstructs binned_features,
    builds the graph, and runs factorization.

    :param snapshot_id: The ID of the snapshot (integer).
    :param graph_df: The full DataFrame with all edges and 'snapshots' column.
    :param motif_fact_args: Arguments for motif factorization (includes clustering method).
    :param temp_motif_roles_dir: Directory where temp motif roles are saved.
    :return: A dictionary {node_id: factorized_role_label_str} for this snapshot.
             Returns an empty dictionary if processing fails or no data.
    """
    print(f"Processing snapshot {snapshot_id} for factorization ({motif_fact_args.cluster_method})...")

    snapshot_edges_df = graph_df[graph_df['snapshots'] <= snapshot_id]

    if snapshot_edges_df.empty:
        print(f"No edges found up to snapshot {snapshot_id}. Skipping.")
        return {}

    nodes_in_snapshot = set(snapshot_edges_df['u'].unique()) | set(snapshot_edges_df['i'].unique())
    if not nodes_in_snapshot:
        print(f"No nodes found in edges up to snapshot {snapshot_id}. Skipping.")
        return {}

    snapshot_graph = nx.Graph()
    snapshot_graph.add_edges_from(snapshot_edges_df[['u', 'i']].values.tolist())
    snapshot_graph.remove_edges_from(nx.selfloop_edges(snapshot_graph))

    if snapshot_graph.number_of_nodes() == 0:
         print(f"No nodes in graph for snapshot {snapshot_id} after removing self-loops. Skipping.")
         return {}

    filename = f'snapshot_{snapshot_id:04d}.pkl'
    file_path = os.path.join(temp_motif_roles_dir, filename)

    if not os.path.exists(file_path):
        print(f"Error: Inverted roles file not found for snapshot {snapshot_id}: {file_path}. Cannot proceed.")
        return {}

    try:
        with open(file_path, 'rb') as f:
            inverted_roles_this_snapshot = pickle.load(f)
    except Exception as e:
        print(f"Error loading inverted roles from {file_path}: {e}. Cannot proceed for this snapshot.")
        return {}

    reconstructed_binned_features = {}
    snapshot_graph_nodes = set(snapshot_graph.nodes())
    nodes_with_features_in_file = set()

    for concatenated_role_str, node_ids_list in inverted_roles_this_snapshot.items():
        individual_feature_strings = concatenated_role_str.split('_') if '_' in concatenated_role_str else [concatenated_role_str]

        for node_id in node_ids_list:
            if node_id in snapshot_graph_nodes:
                reconstructed_binned_features[str(node_id)] = individual_feature_strings
                nodes_with_features_in_file.add(node_id)

    if len(reconstructed_binned_features) < 2:
        print(f"Warning: Only {len(reconstructed_binned_features)} nodes have motif features in snapshot {snapshot_id} (need >= 2). Skipping factorization for this snapshot.")
        return {str(node): "no_features" for node in snapshot_graph_nodes}

    mock_machine = MockMotifMachine(snapshot_graph, reconstructed_binned_features, motif_fact_args)

    try:
        factorization_roles_raw, n_clusters = mock_machine.factorize_string_matrix()
    except Exception as e:
        print(f"Error during factorization for snapshot {snapshot_id}: {e}.")
        return {str(node): "factorization_failed" for node in snapshot_graph_nodes}

    factorization_roles = {}
    for node_str, role_label_str in factorization_roles_raw.items():
        node_id = np.longlong(node_str)
        factorization_roles[node_id] = role_label_str

    return factorization_roles, n_clusters


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factorize motif roles for each snapshot")
    parser.add_argument('--dataset', type=str, default='CollegeMsg', help='Dataset name')
    parser.add_argument('--cluster-method', type=str, default='node_based', choices=['node_based', 'auto', 'max'],
                        help="Method to determine n_clusters: 'node_based' (num_nodes//100) or 'auto' (silhouette score). Default: node_based")
    parser.add_argument('--clusters-max-search', type=int, default=100,
                        help="Maximum number of clusters to search for in 'auto' method. Default: 100")

    args = parser.parse_args()

    grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    temp_motif_roles_dir = os.path.join(grandparent_dir, 'output', args.dataset, 'temp_motif_roles')

    data_snapshots_num = {
        'bitcoinalpha': 274, 'bitcoinotc': 279, 'CollegeMsg': 29,
        'reddit-body': 178, 'reddit-title': 178, 'mathoverflow': 2350,
        'email-Eu-core': 803
    }

    MOTIF_FACT_DEFAULTS = {
        "graphlet_size": 4,
        "quantiles": 5,
        "motif_compression": "factorization",
        "factors": 8,
        "clusters": 50, # Default if node_based calculation isn't possible
        "beta": 0.01,
        "seed": 42
    }
    for key, value in MOTIF_FACT_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, value)

    val_ratio = 0.1
    test_ratio = 0.1

    num_snapshots = data_snapshots_num.get(args.dataset)
    if num_snapshots is None:
        print(f"Error: Dataset '{args.dataset}' not found in data_snapshots_num dictionary. Exiting.")
        exit(1)

    temp_dir_path = os.path.abspath(temp_motif_roles_dir)
    parent_dir_path = os.path.dirname(temp_dir_path)

    if not os.path.exists(temp_dir_path):
        print(f"Error: Temp motif roles directory not found: {temp_dir_path}. Please run the step to generate temporary motif roles first. Exiting.")
        exit(1)

    temp_files = [f for f in os.listdir(temp_dir_path) if re.match(r'snapshot_\d{4}\.pkl', f)]
    num_temp_files = len(temp_files)

    if num_temp_files == 0:
        print(f"Error: No temporary motif role files found in {temp_dir_path}. Cannot perform factorization. Exiting.")
        exit(1)

    if num_temp_files != num_snapshots:
         print(f"Warning: Found {num_temp_files} temp files, but dataset config expects {num_snapshots} snapshots. Processing the {num_temp_files} available files.")
    num_snapshots_to_process = num_temp_files

    output_filename = f'merged_snapshot_factorized_roles_ts{num_snapshots_to_process}_method-{args.cluster_method}_maxk-{args.clusters_max_search}.pkl'
    output_file_path = os.path.join(parent_dir_path, output_filename)

    print("Loading data to get graph structure for snapshot building...")
    try:
        _, _, _, _, _, _, _, _, _, graph_df = \
            get_link_prediction_data(args.dataset, val_ratio, test_ratio, num_snapshots)

        if graph_df is None or graph_df.empty:
             print("Error: Data loading failed or returned empty dataframe. Cannot proceed. Exiting.")
             exit(1)

        if not all(col in graph_df.columns for col in ['u', 'i', 'ts', 'snapshots']):
             raise ValueError("Loaded graph_df is missing required columns ('u', 'i', 'ts', 'snapshots'). Cannot proceed.")

    except Exception as e:
        print(f"Critical Error during graph data loading: {e}")
        print("Cannot proceed without graph structure information. Exiting.")
        exit(1)

    all_snapshots_factorized_roles = {}

    print(f"\nStarting factorization for {num_snapshots_to_process} snapshots using method '{args.cluster_method}'...")

    for snapshot_id in range(1, num_snapshots_to_process + 1):
         snapshot_factorized_roles, n_clusters = process_snapshot_for_factorization(
             snapshot_id,
             graph_df,
             args,
             temp_motif_roles_dir
         )
         all_snapshots_factorized_roles[snapshot_id] = snapshot_factorized_roles
         
         print(f"Snapshot {snapshot_id} processed with {len(snapshot_factorized_roles)} nodes and {n_clusters} clusters.")

    print(f"\nFinished processing all snapshots. Saving merged factorized data to: {output_file_path}")
    try:
        with open(output_file_path, 'wb') as f:
            pickle.dump(all_snapshots_factorized_roles, f)
        print("Merged factorized data saved successfully.")

    except Exception as e:
        print(f"Error saving merged factorized data: {e}")