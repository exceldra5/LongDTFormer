from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import pandas as pd
import pickle
import os
import sys


class Data:
    def __init__(self, 
                 src_node_ids: np.ndarray, 
                 dst_node_ids: np.ndarray, 
                 node_interact_times: np.ndarray, 
                 edge_ids: np.ndarray, 
                 labels: np.ndarray, 
                 snapshots: np.ndarray,
                 u_roles: np.ndarray = None,
                 i_roles: np.ndarray = None):
        self.src_node_ids = src_node_ids
        self.dst_node_ids = dst_node_ids
        self.node_interact_times = node_interact_times
        self.edge_ids = edge_ids
        self.labels = labels
        self.num_interactions = len(src_node_ids)
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)
        self.snapshots = snapshots
        self.u_roles = u_roles
        self.i_roles = i_roles


class CustomizedDataset(Dataset):
    def __init__(self, indices_list: list):
        """
        Customized dataset.
        :param indices_list: list, list of indices
        """
        super(CustomizedDataset, self).__init__()

        self.indices_list = indices_list

    def __getitem__(self, idx: int):
        """
        get item at the index in self.indices_list
        :param idx: int, the index
        :return:
        """
        return self.indices_list[idx]

    def __len__(self):
        return len(self.indices_list)


def get_idx_data_loader(indices_list: list, batch_size: int, shuffle: bool):
    """
    get data loader that iterates over indices
    :param indices_list: list, list of indices
    :param batch_size: int, batch size
    :param shuffle: boolean, whether to shuffle the data
    :return: data_loader, DataLoader
    """
    dataset = CustomizedDataset(indices_list=indices_list)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=False)
    return data_loader


# role aware data loader
def get_role_idx_data_loader(
    train_data_indices: list,
    # batch_src_node_ids: np.ndarray, 
    # batch_src_node_snapshots: np.ndarray,
    # batch_src_node_roles: np.ndarray,
    train_data: Data, 
    graph_df: pd.DataFrame,
    batch_size: int):
        
    # select batch_size indices from train_data_indices
    selected_indices = np.random.choice(train_data_indices, size=batch_size, replace=False)
    batch_src_node_ids = train_data.src_node_ids[selected_indices]
    batch_src_node_snapshots = train_data.snapshots[selected_indices]
    batch_src_node_roles = train_data.u_roles[selected_indices]

    # get the node ids in the same snapshot with the same role
    batch_src_same_role_node_ids = []
    for i in range(batch_size):
        src_node_id = batch_src_node_ids[i]
        src_node_snapshot = batch_src_node_snapshots[i]
        src_node_role = batch_src_node_roles[i]
        # get the node ids in the same snapshot with the same role
        src_same_role_df = graph_df[(graph_df['snapshots'] == src_node_snapshot) & (graph_df['u_role'] == src_node_role)]
        src_same_role_node_ids = src_same_role_df['u'].values
        
        # select a random node id from the same snapshot with the same role
        if len(src_same_role_node_ids) > 0:
            batch_src_same_role_node_ids.append(np.random.choice(src_same_role_node_ids, size=1, replace=False)[0])
        else:
            batch_src_same_role_node_ids.append(src_node_id) # NOTE(wsgwak): is it okay?
            
    return batch_src_node_ids, batch_src_node_snapshots, batch_src_node_roles, batch_src_same_role_node_ids, selected_indices


def get_link_prediction_data_with_roles(dataset_name: str, val_ratio: float, test_ratio: float, num_snapshots: int):
    """
    generate data for link prediction task (inductive & transductive settings)
    :param dataset_name: str, dataset name
    :param val_ratio: float, validation data ratio
    :param test_ratio: float, test data ratio
    :return: node_raw_features, edge_raw_features, (np.ndarray),
            full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, (Data object)
    """
    grandparent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load data and train val test split
    graph_df_path = os.path.join(grandparent_dir, 'processed_data', dataset_name, f'ml_{dataset_name}.csv')
    graph_df = pd.read_csv(graph_df_path)
    edge_raw_features_path = os.path.join(grandparent_dir, 'processed_data', dataset_name, f'ml_{dataset_name}.npy')
    edge_raw_features = np.load(edge_raw_features_path)
    node_raw_features_path = os.path.join(grandparent_dir, 'processed_data', dataset_name, f'ml_{dataset_name}_node.npy')
    node_raw_features = np.load(node_raw_features_path)

    # Load node role data
    # node_role_data_path = f'./output/{dataset_name}/merged_snapshot_factorized_roles.pkl'
    node_role_data_path = os.path.join(grandparent_dir, 'output', dataset_name, 'merged_snapshot_factorized_roles.pkl')
    if not os.path.exists(node_role_data_path):
        print(f"File {node_role_data_path} does not exist.")
        return
    with open(node_role_data_path, "rb") as f:
        node_role_data = pickle.load(f)

    # Process node_role_data into a DataFrame
    records = []
    for time_step, node_roles in node_role_data.items():
        for node_id, role in node_roles.items():
            records.append({
                "snapshots": time_step,
                "node_id": int(node_id),  # ensure consistent dtype
                "role": role
            })
    node_role_data_pd = pd.DataFrame(records)

    # # Add roles to graph_df by merging on both u and i separately
    # graph_df = graph_df.copy()
    # graph_df = graph_df.merge(node_role_data_pd.rename(columns={"node_id": "u", "role": "u_role"}), on=["snapshots", "u"], how="left")
    # graph_df = graph_df.merge(node_role_data_pd.rename(columns={"node_id": "i", "role": "i_role"}), on=["snapshots", "i"], how="left")


    # preprocessing
    NODE_FEAT_DIM = EDGE_FEAT_DIM = 172
    assert NODE_FEAT_DIM >= node_raw_features.shape[1], f'Node feature dimension in dataset {dataset_name} is bigger than {NODE_FEAT_DIM}!'
    assert EDGE_FEAT_DIM >= edge_raw_features.shape[1], f'Edge feature dimension in dataset {dataset_name} is bigger than {EDGE_FEAT_DIM}!'
    # padding the features of edges and nodes to the same dimension (172 for all the datasets)
    if node_raw_features.shape[1] < NODE_FEAT_DIM:
        node_zero_padding = np.zeros((node_raw_features.shape[0], NODE_FEAT_DIM - node_raw_features.shape[1]))
        node_raw_features = np.concatenate([node_raw_features, node_zero_padding], axis=1)
    if edge_raw_features.shape[1] < EDGE_FEAT_DIM:
        edge_zero_padding = np.zeros((edge_raw_features.shape[0], EDGE_FEAT_DIM - edge_raw_features.shape[1]))
        edge_raw_features = np.concatenate([edge_raw_features, edge_zero_padding], axis=1)

    assert NODE_FEAT_DIM == node_raw_features.shape[1] and EDGE_FEAT_DIM == edge_raw_features.shape[1], 'Unaligned feature dimensions after feature padding!'

    # get the timestamp of validate and test set
    val_time, test_time = list(np.quantile(graph_df.ts, [(1 - val_ratio - test_ratio), (1 - test_ratio)]))

    min_ts = graph_df['ts'].min()
    max_ts = graph_df['ts'].max()
    range_size = (max_ts - min_ts) / num_snapshots

    # snapshots column starts from 1
    graph_df['snapshots'] = ((graph_df['ts'] - min_ts) / range_size).astype(np.int16) + 1

    graph_df.loc[graph_df['snapshots'] >= num_snapshots, 'snapshots'] = num_snapshots
    
    # Add roles to graph_df by merging on both u and i separately
    # graph_df = graph_df.copy()
    graph_df = graph_df.merge(node_role_data_pd.rename(columns={"node_id": "u", "role": "u_role"}), on=["snapshots", "u"], how="left")
    graph_df = graph_df.merge(node_role_data_pd.rename(columns={"node_id": "i", "role": "i_role"}), on=["snapshots", "i"], how="left")


    src_node_ids = graph_df.u.values.astype(np.longlong)
    dst_node_ids = graph_df.i.values.astype(np.longlong)
    node_interact_times = graph_df.ts.values.astype(np.float64)
    edge_ids = graph_df.idx.values.astype(np.longlong)
    labels = graph_df.label.values
    snapshots = graph_df.snapshots.values

    df_long = pd.concat([
        graph_df.rename(columns={'u': 'node'})[['node', 'snapshots']],
        graph_df.rename(columns={'i': 'node'})[['node', 'snapshots']]
    ])

    # count node appearance in each snapshot
    node_counts_per_snapshot = df_long.groupby(['node', 'snapshots']).size().unstack(fill_value=0)

    all_nodes = np.sort(np.unique(graph_df[['u', 'i']].values))
    # all_snapshots = np.sort(graph_df['snapshots'].unique())
    all_snapshots = np.arange(1, num_snapshots + 1)

    node_counts_per_snapshot = node_counts_per_snapshot.reindex(index=all_nodes, columns=all_snapshots, fill_value=0)

    node_snap_counts = node_counts_per_snapshot.values

    zero_vector = np.zeros((1, node_snap_counts.shape[1]))

    node_snap_counts = np.vstack([zero_vector, node_snap_counts])

    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids, node_interact_times=node_interact_times, edge_ids=edge_ids, labels=labels, snapshots=snapshots)

    # the setting of seed follows previous works
    random.seed(2020)

    # union to get node set
    node_set = set(src_node_ids) | set(dst_node_ids)
    num_total_unique_node_ids = len(node_set)

    train_mask = (node_interact_times <= val_time)

    u_roles = graph_df.u_role.values
    i_roles = graph_df.i_role.values
    train_data = Data(
        src_node_ids=src_node_ids[train_mask],
        dst_node_ids=dst_node_ids[train_mask],
        node_interact_times=node_interact_times[train_mask],
        edge_ids=edge_ids[train_mask],
        labels=labels[train_mask],
        snapshots=snapshots[train_mask],
        u_roles=u_roles[train_mask],
        i_roles=i_roles[train_mask]
    )

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # validation and test data
    val_data = Data(
        src_node_ids=src_node_ids[val_mask],
        dst_node_ids=dst_node_ids[val_mask],
        node_interact_times=node_interact_times[val_mask],
        edge_ids=edge_ids[val_mask],
        labels=labels[val_mask],
        snapshots=snapshots[val_mask],
        u_roles=u_roles[val_mask],
        i_roles=i_roles[val_mask]
    )
    
    test_data = Data(
        src_node_ids=src_node_ids[test_mask],
        dst_node_ids=dst_node_ids[test_mask],
        node_interact_times=node_interact_times[test_mask],
        edge_ids=edge_ids[test_mask],
        labels=labels[test_mask],
        snapshots=snapshots[test_mask],
        u_roles=u_roles[test_mask],
        i_roles=i_roles[test_mask]
    )


    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, None, None, node_snap_counts, graph_df


if __name__ == "__main__":
    dataset_name = 'CollegeMsg'
    val_ratio = 0.1
    test_ratio = 0.1
    num_snapshots = 29

    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, new_node_val_data, new_node_test_data, node_snap_counts, graph_df = get_link_prediction_data_with_roles(
        dataset_name=dataset_name,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        num_snapshots=num_snapshots
    )