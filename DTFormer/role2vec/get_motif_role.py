import pandas as pd
import numpy as np
import networkx as nx
import random # Assuming this import is needed for your original loader
import argparse # To simulate the args object for motif counting
import pickle   # For saving the output
import os       # For creating directories

# Assuming the Role2Vec files (motif_count.py, utils.py, param_parser.py) are in a 'src' directory
# If they are in the same directory as this script, you might need to adjust imports
try:
    from motif_count import MotifCounterMachine
    # from utils import tab_printer, load_graph, create_documents # Import utils functions just in case
    # We only need parts of param_parser for motif args, will simulate below
except ImportError:
     print("Warning: Could not import Role2Vec components from src/. Using mock classes.")
     print("Please ensure src/motif_count.py and src/utils.py are accessible.")
     exit(1) # Exit if imports fail


# --- Your provided data loading function (with minor error handling/fixes) ---
# (Assuming Data class is defined elsewhere or is the minimal mock above)
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
    graph_df_path = '../processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name)
    edge_feat_path = '../processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name)
    node_feat_path = '../processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name)

    try:
        graph_df = pd.read_csv(graph_df_path)
        edge_raw_features = np.load(edge_feat_path)
        node_raw_features = np.load(node_feat_path)
        print(f"Successfully loaded data for dataset: {dataset_name}")
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Attempted paths: {graph_df_path}, {edge_feat_path}, {node_feat_path}")
        print("Returning dummy data for demonstration. Please ensure dataset files exist for real use.")
        # Return dummy data or raise error if files are not found
        num_edges = 100
        num_nodes = 50
        dummy_df = pd.DataFrame({
            'u': np.random.randint(0, num_nodes, num_edges),
            'i': np.random.randint(0, num_nodes, num_edges),
            'ts': np.sort(np.random.rand(num_edges) * 1000),
            'idx': np.arange(num_edges),
            'label': np.random.randint(0, 2, num_edges)
        })
        dummy_node_feat = np.random.rand(num_nodes + 1, 10) # +1 for potential node 0
        dummy_edge_feat = np.random.rand(num_edges, 10)
        graph_df = dummy_df
        node_raw_features = dummy_node_feat
        edge_raw_features = dummy_edge_feat
    except Exception as e:
         print(f"An unexpected error occurred loading data: {e}")
         raise # Re-raise the exception after printing


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

# --- Function to get motif roles per snapshot (UPDATED) ---

def get_motif_roles_per_snapshot(graph_df: pd.DataFrame, num_snapshots: int, motif_args: argparse.Namespace, output_path: str):
    """
    Generates motif-based roles for each node in each time snapshot,
    stores them in the format snapshot_id -> role_id -> list of node_ids,
    and saves the result to a pickle file.

    :param graph_df: DataFrame containing all edges with 'u', 'i', and 'snapshots' columns.
    :param num_snapshots: Total number of snapshots.
    :param motif_args: An object/Namespace containing motif counting parameters
                       (graphlet_size, quantiles, motif_compression, factors, clusters, beta, seed).
    :param output_path: The file path to save the pickled roles dictionary.
    :return: A dictionary where keys are snapshot IDs (1 to num_snapshots) and
             values are dictionaries mapping role IDs (str) to lists of node IDs (longlong).
    """
    # Dictionary to store roles in the desired format: snapshot_id -> role_id -> [node_ids]
    all_snapshot_roles_inverted = {}

    print(f"\nGenerating motif roles for {num_snapshots} snapshots...")

    for snapshot_id in range(1, num_snapshots + 1):
        print(f"Processing snapshot {snapshot_id}/{num_snapshots}...")

        # Filter edges for the current snapshot (cumulative up to this point)
        snapshot_edges_df = graph_df[graph_df['snapshots'] <= snapshot_id]

        if snapshot_edges_df.empty:
            print(f"No edges found up to snapshot {snapshot_id}. Skipping.")
            all_snapshot_roles_inverted[snapshot_id] = {} # Store empty dict for this snapshot
            continue

        # Build NetworkX graph for the current snapshot
        # Use u and i columns to create edgelist
        # Ensure nodes are correctly represented (NetworkX usually handles mixed types but explicit might be safer if issues arise)
        snapshot_graph = nx.from_edgelist(snapshot_edges_df[['u', 'i']].values.tolist())

        # Remove self-loops as Role2Vec's load_graph does
        snapshot_graph.remove_edges_from(nx.selfloop_edges(snapshot_graph))

        # Check if the graph is still empty or has no nodes after removing self-loops
        if not snapshot_graph.nodes():
             print(f"Graph for snapshot {snapshot_id} has no nodes after processing. Skipping.")
             all_snapshot_roles_inverted[snapshot_id] = {}
             continue

        # Instantiate MotifCounterMachine
        try:
            motif_machine = MotifCounterMachine(snapshot_graph, motif_args)
        except Exception as e:
            print(f"Error initializing MotifCounterMachine for snapshot {snapshot_id}: {e}")
            all_snapshot_roles_inverted[snapshot_id] = {}
            continue

        # Calculate motif features and get labels
        try:
            # This returns {node_id_str: [label_str]} or {node_id_str: label_str}
            snapshot_roles_raw = motif_machine.create_string_labels()
        except Exception as e:
            print(f"Error calculating motif roles for snapshot {snapshot_id}: {e}")
            all_snapshot_roles_inverted[snapshot_id] = {}
            continue

        # --- Restructure the roles into the desired format ---
        # Original format: {node_id_str: label_val (list or str)}
        # Desired format: {role_id_str: [node_ids (longlong)]}
        current_snapshot_inverted_roles = {}
        for node_str, label_val in snapshot_roles_raw.items():
            # Convert node ID back to original type (e.g., longlong)
            node_id = np.longlong(node_str)
            # Get the actual role label string
            role_label_str = label_val[0] if isinstance(label_val, list) else label_val
            if isinstance(label_val, list):
                if len(label_val) > 1:
                    print(f"Warning: Node {node_str} has multiple labels: {label_val}. Using first label '{role_label_str}'.")
                    exit(1) # Exit if multiple labels found, as per your original code

            # Add node ID to the list for this role label in the inverted dictionary
            if role_label_str not in current_snapshot_inverted_roles:
                current_snapshot_inverted_roles[role_label_str] = []
            current_snapshot_inverted_roles[role_label_str].append(node_id)

        all_snapshot_roles_inverted[snapshot_id] = current_snapshot_inverted_roles

    # --- Save the result to a pickle file ---
    print(f"\nSaving motif roles to {output_path}...")
    output_dir = os.path.dirname(output_path)
    if output_dir: # Only try to create directory if path includes one
        os.makedirs(output_dir, exist_ok=True)

    try:
        with open(output_path, 'wb') as f:
            pickle.dump(all_snapshot_roles_inverted, f)
        print("Save complete.")
    except Exception as e:
        print(f"Error saving pickle file to {output_path}: {e}")


    return all_snapshot_roles_inverted


def test():
    # Define dataset parameters
    dataset_name = "dummy" # Replace with your dataset name like "wikipedia" or "reddit"
    val_ratio = 0.15
    test_ratio = 0.15
    num_snapshots = 10 # Define the number of snapshots you want

    # Define the output path for the pickle file
    output_file_path = f"./output/{dataset_name}_motif_roles_{num_snapshots}_snapshots.pkl"


    # Simulate the motif-specific arguments needed by MotifCounterMachine
    # These values are defaults from src/param_parser.py
    motif_args = argparse.Namespace()
    motif_args.graphlet_size = 4
    motif_args.quantiles = 5
    motif_args.motif_compression = "string" # or "factorization"
    motif_args.factors = 8
    motif_args.clusters = 50
    motif_args.beta = 0.01
    motif_args.seed = 42 # Used by KMeans/NMF in factorization

    # Step 1: Load your data using your function
    # This function call might print file not found errors if './processed_data/{dataset_name}/ml_{dataset_name}.csv' etc. don't exist.
    # The code includes dummy data creation in case of FileNotFoundError for demonstration.
    print("Loading data...")
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _, node_snap_counts, graph_df = \
        get_link_prediction_data(dataset_name, val_ratio, test_ratio, num_snapshots)

    print("\nData Loading Complete.")
    # print(f"Loaded {len(graph_df)} total edges.")
    # print(f"Snapshot distribution:\n{graph_df['snapshots'].value_counts().sort_index()}")


    # Step 2: Get motif-based roles for each snapshot and save to pickle
    # Pass the dataframe with snapshot IDs, motif arguments, and the output path
    snapshot_motif_roles_inverted = get_motif_roles_per_snapshot(graph_df, num_snapshots, motif_args, output_file_path)

    # Step 3: Verify the output structure and contents (optional)
    print("\nVerification of loaded roles (first snapshot, first 5 roles):")
    try:
        with open(output_file_path, 'rb') as f:
            loaded_roles = pickle.load(f)

        snapshot_id_to_check = 1
        if snapshot_id_to_check in loaded_roles:
            roles_snap_check = loaded_roles[snapshot_id_to_check]
            print(f"Loaded roles for snapshot {snapshot_id_to_check}:")
            count = 0
            if roles_snap_check:
                for role_id, node_list in roles_snap_check.items():
                    print(f"  Role '{role_id}': {len(node_list)} nodes -> {node_list[:5]}...") # Print first 5 nodes
                    count += 1
                    if count >= 5: # Print only 5 roles
                        break
            else:
                print(f"  Snapshot {snapshot_id_to_check} has no roles.")
        else:
            print(f"Snapshot {snapshot_id_to_check} not found in loaded roles.")

        # Verify last snapshot
        snapshot_id_to_check = num_snapshots
        if snapshot_id_to_check in loaded_roles:
            roles_snap_check = loaded_roles[snapshot_id_to_check]
            print(f"Loaded roles for snapshot {snapshot_id_to_check} (Last Snapshot):")
            count = 0
            if roles_snap_check:
                 for role_id, node_list in roles_snap_check.items():
                    print(f"  Role '{role_id}': {len(node_list)} nodes -> {node_list[:5]}...") # Print first 5 nodes
                    count += 1
                    if count >= 5: # Print only 5 roles
                        break
            else:
                 print(f"  Snapshot {snapshot_id_to_check} has no roles.")
        else:
             print(f"Snapshot {snapshot_id_to_check} not found in loaded roles.")


    except FileNotFoundError:
        print(f"Could not load {output_file_path} for verification.")
    except Exception as e:
        print(f"Error during verification: {e}")


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="Get Motif Roles")
    parser.add_argument('--dataset', type=str, default='bitcoinotc', help='dataset name')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='validation data ratio')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='test data ratio')
    parser.add_argument('--num_motifs', type=int, default=4, help='number of motifs')
    args = parser.parse_args()
    
    data_snapshots_num = {'bitcoinalpha': 274,
                          'bitcoinotc': 279,
                          'CollegeMsg': 29,
                          'reddit-body': 178,
                          'reddit-title': 178,
                          'mathoverflow': 2350,
                          'email-Eu-core': 803}
    num_snapshots = data_snapshots_num[args.dataset] if args.dataset in data_snapshots_num else 10
    
    args.graphlet_size = 4
    args.quantiles = 5
    args.motif_compression = "string" # or "factorization"
    args.factors = 8
    args.clusters = 50
    args.beta = 0.01
    args.seed = 42 # Used by KMeans/NMF in factorization
    
    output_file_path = f"./output/{args.dataset}_motif_roles_{num_snapshots}_snapshots.pkl"
    
    # Step 1: Load your data using your function
    # This function call might print file not found errors if './processed_data/{dataset_name}/ml_{dataset_name}.csv' etc. don't exist.
    # The code includes dummy data creation in case of FileNotFoundError for demonstration.
    print("Loading data...")
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _, node_snap_counts, graph_df = \
        get_link_prediction_data(args.dataset, args.val_ratio, args.test_ratio, num_snapshots)

    print("\nData Loading Complete.")
    # print(f"Loaded {len(graph_df)} total edges.")
    # print(f"Snapshot distribution:\n{graph_df['snapshots'].value_counts().sort_index()}")


    # Step 2: Get motif-based roles for each snapshot and save to pickle
    # Pass the dataframe with snapshot IDs, motif arguments, and the output path
    snapshot_motif_roles_inverted = get_motif_roles_per_snapshot(graph_df, num_snapshots, args, output_file_path)

    # Step 3: Verify the output structure and contents (optional)
    print("\nVerification of loaded roles (first snapshot, first 5 roles):")
    try:
        with open(output_file_path, 'rb') as f:
            loaded_roles = pickle.load(f)

        snapshot_id_to_check = 1
        if snapshot_id_to_check in loaded_roles:
            roles_snap_check = loaded_roles[snapshot_id_to_check]
            print(f"Loaded roles for snapshot {snapshot_id_to_check}:")
            count = 0
            if roles_snap_check:
                for role_id, node_list in roles_snap_check.items():
                    print(f"  Role '{role_id}': {len(node_list)} nodes -> {node_list[:5]}...") # Print first 5 nodes
                    count += 1
                    if count >= 5: # Print only 5 roles
                        break
            else:
                print(f"  Snapshot {snapshot_id_to_check} has no roles.")
        else:
            print(f"Snapshot {snapshot_id_to_check} not found in loaded roles.")

        # Verify last snapshot
        snapshot_id_to_check = num_snapshots
        if snapshot_id_to_check in loaded_roles:
            roles_snap_check = loaded_roles[snapshot_id_to_check]
            print(f"Loaded roles for snapshot {snapshot_id_to_check} (Last Snapshot):")
            count = 0
            if roles_snap_check:
                 for role_id, node_list in roles_snap_check.items():
                    print(f"  Role '{role_id}': {len(node_list)} nodes -> {node_list[:5]}...") # Print first 5 nodes
                    count += 1
                    if count >= 5: # Print only 5 roles
                        break
            else:
                 print(f"  Snapshot {snapshot_id_to_check} has no roles.")
        else:
             print(f"Snapshot {snapshot_id_to_check} not found in loaded roles.")


    except FileNotFoundError:
        print(f"Could not load {output_file_path} for verification.")
    except Exception as e:
        print(f"Error during verification: {e}")