import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
import os
from collections import defaultdict

# Define the expected feature dimensions from your project
# You might need to adjust these if they differ for your actual datasets
NODE_FEAT_DIM = 172
EDGE_FEAT_DIM = 172 # RTGCN DBLP doesn't have edge features, we'll return None

# Simulate the Data class structure for compatibility, or return dictionaries
# If your Data class has specific methods, you might need to implement them here
# or adapt your code to use dictionaries instead.
class Data:
    def __init__(self, src_node_ids, dst_node_ids, node_interact_times, edge_ids, labels, snapshots):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels # Note: These will be edge labels (e.g., 1 for existing edge), NOT RTGCN's node classification labels
        self.snapshots = snapshots

        self.num_interactions = len(src_node_ids)
        # Assume node IDs are 0-based and contiguous for simplicity based on DBLP data structure
        self.num_unique_nodes = max(max(src_node_ids) if len(src_node_ids) > 0 else -1,
                                    max(dst_node_ids) if len(dst_node_ids) > 0 else -1) + 1 if len(src_node_ids) > 0 or len(dst_node_ids) > 0 else 0


def load_rtgcn_data_for_your_project(dataset_path: str, val_ratio: float, test_ratio: float, num_snapshots: int):
    """
    Loads RTGCN-formatted data (e.g., DBLP3.npz) and converts it
    into a format similar to the user's link prediction data loading function.

    Args:
        dataset_path (str): Path to the RTGCN .npz file (e.g., './data/DBLP3/DBLP3.npz').
        val_ratio (float): Ratio for validation data split (based on snapshots).
        test_ratio (float): Ratio for test data split (based on snapshots).
                            The split is by snapshot index:
                            Train: first N snapshots according to (1-val_ratio-test_ratio)
                            Val: next M snapshots according to val_ratio
                            Test: remaining snapshots according to test_ratio
        num_snapshots (int): The number of snapshots to consider from the data.
                             RTGCN's NPZ file might have more, we take the first `num_snapshots`.

    Returns:
        tuple: (node_raw_features, edge_raw_features, full_data, train_data,
                val_data, test_data, new_node_val_data, new_node_test_data, node_snap_counts)
               Note: edge_raw_features, new_node_val_data, new_node_test_data will be None.
               Labels in Data objects will be 1 for all edges reconstructed from adjs.
               Node labels from RTGCN's 'labels' in npz are *not* included in Data objects.
    """
    try:
        data_npz = np.load(dataset_path, allow_pickle=True)
    except FileNotFoundError:
        print(f"Error: Dataset file not found at {dataset_path}")
        return None

    adjs = data_npz['adjs']  # Shape (num_total_snaps, num_nodes, num_nodes)
    attmats = data_npz['attmats']  # Shape (num_nodes, num_total_snaps, feature_dim)
    # labels_node = data_npz.get('labels', None) # Node labels, not used in Data objects for edge prediction

    num_total_snaps = adjs.shape[0]
    num_nodes = adjs.shape[1]
    feature_dim = attmats.shape[2]

    if num_snapshots > num_total_snaps:
        print(f"Warning: Requested {num_snapshots} snapshots, but data only has {num_total_snaps}. Using {num_total_snaps}.")
        num_snapshots = num_total_snaps

    # 1. Reconstruct Interaction Data from Adjacency Matrices
    # We'll treat each edge present in each snapshot as an interaction.
    # Assign snapshot index as a proxy for timestamp.
    # Assign a dummy edge_id and a dummy label (1 for existing edge).

    all_src_node_ids = []
    all_dst_node_ids = []
    all_node_interact_times = [] # Using snapshot index as time proxy
    all_edge_ids = []
    all_labels = [] # Dummy label for existing edges
    all_snapshots = []

    global_edge_idx_counter = 0
    node_activity_counts = defaultdict(lambda: defaultdict(int)) # {node_id: {snapshot_id: count}}

    for snap_idx in range(num_snapshots):
        adj_snap = adjs[snap_idx]
        # Convert to COO for easy iteration over non-zero elements
        adj_coo = sp.coo_matrix(adj_snap)

        for r, c, v in zip(adj_coo.row, adj_coo.col, adj_coo.data):
            # For undirected graph, only add each edge once (e.g., r <= c) if needed,
            # but here we just take all non-zero entries which includes both (r,c) and (c,r)
            # This might result in more interactions than unique edges per snapshot,
            # but captures the 'presence' of the connection.
            # If you strictly need unique (u,i) pairs per snapshot, add a check.
            # Let's add the (r,c) and (c,r) if they exist, as it reflects the matrix structure.

            all_src_node_ids.append(r)
            all_dst_node_ids.append(c)
            all_node_interact_times.append(snap_idx) # Use snapshot index as time
            all_edge_ids.append(global_edge_idx_counter)
            all_labels.append(1) # Label 1 for edge presence
            all_snapshots.append(snap_idx)

            node_activity_counts[r][snap_idx] += 1
            node_activity_counts[c][snap_idx] += 1 # Count both ends

            global_edge_idx_counter += 1

    # Convert lists to numpy arrays
    src_node_ids = np.array(all_src_node_ids, dtype=np.longlong)
    dst_node_ids = np.array(all_dst_node_ids, dtype=np.longlong)
    node_interact_times = np.array(all_node_interact_times, dtype=np.float64) # Use float type as in your code
    edge_ids = np.array(all_edge_ids, dtype=np.longlong)
    labels = np.array(all_labels, dtype=np.int16)
    snapshots = np.array(all_snapshots, dtype=np.int16)

    # 2. Prepare Node Features
    # RTGCN has time-varying features. Your format expects static node_raw_features.
    # Let's take features from the first snapshot (time 0) as the static features.
    # You might consider using features from the last snapshot or averaging if more suitable.
    node_raw_features = attmats[:, 0, :] # Shape (num_nodes, feature_dim)

    # Apply padding if necessary, based on your script's logic
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    assert node_raw_features.shape[1] == NODE_FEAT_DIM, "Node feature padding failed"

    # 3. Prepare Edge Features (Not available in RTGCN DBLP)
    edge_raw_features = None # Or generate zeros if your downstream code requires it

    # 4. Compute Node Snapshot Counts
    # This requires knowing all possible nodes and snapshots
    all_nodes = np.unique(np.concatenate([src_node_ids, dst_node_ids]))
    all_snap_indices = np.arange(num_snapshots) # Use 0-based index

    node_counts_per_snapshot_matrix = np.zeros((num_nodes, num_snapshots), dtype=np.int32)

    for node_id in range(num_nodes):
         for snap_idx in range(num_snapshots):
             node_counts_per_snapshot_matrix[node_id, snap_idx] = node_activity_counts[node_id].get(snap_idx, 0)

    # Your code adds a zero vector at the beginning, let's replicate this
    zero_vector = np.zeros((1, num_snapshots), dtype=np.int32)
    node_snap_counts = np.vstack([zero_vector, node_counts_per_snapshot_matrix])


    # 5. Simulate Train/Val/Test Split based on Snapshots
    # Your split logic uses time quantiles. Here, we'll split by snapshot index.
    # This is more aligned with how RTGCN evaluates discrete snapshots.
    # You might need to adjust val_ratio/test_ratio interpretation or your downstream
    # logic if a strict time-quantile split is critical.

    num_total_interactions = len(src_node_ids)

    # Sort interactions by snapshot index (proxy for time)
    sort_indices = np.argsort(snapshots)
    src_node_ids_sorted = src_node_ids[sort_indices]
    dst_node_ids_sorted = dst_node_ids[sort_indices]
    node_interact_times_sorted = node_interact_times[sort_indices]
    edge_ids_sorted = edge_ids[sort_indices]
    labels_sorted = labels[sort_indices]
    snapshots_sorted = snapshots[sort_indices]

    # Find interaction index thresholds based on snapshot quantiles
    # Calculate cumulative counts per snapshot
    snap_counts = pd.Series(snapshots).value_counts().sort_index().cumsum().values
    if len(snap_counts) < num_snapshots:
         # Handle cases where some snapshots might have no edges by padding cumsum
         full_snap_counts = np.zeros(num_snapshots)
         existing_snaps = sorted(list(pd.Series(snapshots).unique()))
         current_idx = 0
         for i in range(num_snapshots):
              if i in existing_snaps:
                   full_snap_counts[i] = snap_counts[existing_snaps.index(i)]
                   current_idx = existing_snaps.index(i) # Keep track of which cumsum value to use
              else:
                   full_snap_counts[i] = snap_counts[current_idx] # Use the last known count
         snap_counts = full_snap_counts


    train_val_test_split_snap = int(num_snapshots * (1 - val_ratio - test_ratio))
    train_test_split_snap = int(num_snapshots * (1 - test_ratio))

    # Get the index in the sorted interaction list corresponding to these snapshot cutoffs
    # Find the index of the first interaction *after* the cutoff snapshot
    train_end_idx = np.searchsorted(snapshots_sorted, train_val_test_split_snap, side='left')
    val_end_idx = np.searchsorted(snapshots_sorted, train_test_split_snap, side='left')


    # Split the sorted data arrays
    train_mask_indices = slice(0, train_end_idx)
    val_mask_indices = slice(train_end_idx, val_end_idx)
    test_mask_indices = slice(val_end_idx, num_total_interactions)


    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                     node_interact_times=node_interact_times, edge_ids=edge_ids,
                     labels=labels, snapshots=snapshots)

    train_data = Data(src_node_ids=src_node_ids_sorted[train_mask_indices],
                      dst_node_ids=dst_node_ids_sorted[train_mask_indices],
                      node_interact_times=node_interact_times_sorted[train_mask_indices],
                      edge_ids=edge_ids_sorted[train_mask_indices],
                      labels=labels_sorted[train_mask_indices],
                      snapshots=snapshots_sorted[train_mask_indices])

    val_data = Data(src_node_ids=src_node_ids_sorted[val_mask_indices],
                    dst_node_ids=dst_node_ids_sorted[val_mask_indices],
                    node_interact_times=node_interact_times_sorted[val_mask_indices],
                    edge_ids=edge_ids_sorted[val_mask_indices],
                    labels=labels_sorted[val_mask_indices],
                    snapshots=snapshots_sorted[val_mask_indices])

    test_data = Data(src_node_ids=src_node_ids_sorted[test_mask_indices],
                     dst_node_ids=dst_node_ids_sorted[test_mask_indices],
                     node_interact_times=node_interact_times_sorted[test_mask_indices],
                     edge_ids=edge_ids_sorted[test_mask_indices],
                     labels=labels_sorted[test_mask_indices],
                     snapshots=snapshots_sorted[test_mask_indices])


    print("\n--- RTGCN Data Loading Summary ---")
    print(f"Loaded data from: {dataset_path}")
    print(f"Number of snapshots used: {num_snapshots}")
    print("--- Reconstructed Interaction Data ---")
    print("Note: Timestamps are snapshot indices. Labels are 1 for all reconstructed edges.")
    print("Node labels from NPZ are NOT included in Data objects.")
    print("Edge features are None.")
    print("------------------------------------")
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))
    print("------------------------------------")


    return (node_raw_features, edge_raw_features, full_data, train_data,
            val_data, test_data, None, None, node_snap_counts)


def test():
    # grand parent dir
    grand_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_file = os.path.join(grand_parent_dir, 'processed_data', 'DBLP3', 'DBLP3.npz')
    # data_file = '../processed_data/DBLP3/DBLP3.npz' # Replace with your actual data path
    num_snaps_to_use = 3 # As used in RTGCN's example

    # Ratios to split 3 snapshots: ~0.33 for each
    # This will roughly assign interactions from snap 0 to train, snap 1 to val, snap 2 to test
    val_r = 1/3
    test_r = 1/3

    (node_raw_features, edge_raw_features, full_data, train_data,
     val_data, test_data, new_node_val_data, new_node_test_data, node_snap_counts) = \
        load_rtgcn_data_for_your_project(data_file, val_r, test_r, num_snaps_to_use)

    if node_raw_features is not None:
        print(f"\nnode_raw_features shape: {node_raw_features.shape}")
        print(f"edge_raw_features is: {edge_raw_features}") # Should be None
        print(f"node_snap_counts shape: {node_snap_counts.shape}")
        print("\nTrain Data Sample (first 5):")
        if train_data.num_interactions > 0:
             print(f"  src_node_ids[:5]: {train_data.src_node_ids[:5]}")
             print(f"  dst_node_ids[:5]: {train_data.dst_node_ids[:5]}")
             print(f"  node_interact_times[:5]: {train_data.node_interact_times[:5]}")
             print(f"  snapshots[:5]: {train_data.snapshots[:5]}")
             print(f"  labels[:5]: {train_data.labels[:5]}")
        else:
             print("  (No train data)")

        print("\nVal Data Sample (first 5):")
        if val_data.num_interactions > 0:
             print(f"  src_node_ids[:5]: {val_data.src_node_ids[:5]}")
             print(f"  dst_node_ids[:5]: {val_data.dst_node_ids[:5]}")
             print(f"  node_interact_times[:5]: {val_data.node_interact_times[:5]}")
             print(f"  snapshots[:5]: {val_data.snapshots[:5]}")
             print(f"  labels[:5]: {val_data.labels[:5]}")
        else:
            print("  (No validation data)")

        print("\nTest Data Sample (first 5):")
        if test_data.num_interactions > 0:
             print(f"  src_node_ids[:5]: {test_data.src_node_ids[:5]}")
             print(f"  dst_node_ids[:5]: {test_data.dst_node_ids[:5]}") # Correction: should be test_data
             print(f"  node_interact_times[:5]: {test_data.node_interact_times[:5]}")
             print(f"  snapshots[:5]: {test_data.snapshots[:5]}")
             print(f"  labels[:5]: {test_data.labels[:5]}")
        else:
            print("  (No test data)")
            
            
if __name__ == '__main__':
    test()
    
    