from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import numpy as np
import random
import pandas as pd


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
    """
    # Load data and train val test split
    graph_df = pd.read_csv('./processed_data/{}/ml_{}.csv'.format(dataset_name, dataset_name))
    edge_raw_features = np.load('./processed_data/{}/ml_{}.npy'.format(dataset_name, dataset_name))
    node_raw_features = np.load('./processed_data/{}/ml_{}_node.npy'.format(dataset_name, dataset_name))

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

    # from 1
    graph_df['snapshots'] = ((graph_df['ts'] - min_ts) / range_size).astype(np.int16) + 1

    graph_df.loc[graph_df['snapshots'] >= num_snapshots, 'snapshots'] = num_snapshots

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

    train_data = Data(src_node_ids=src_node_ids[train_mask], dst_node_ids=dst_node_ids[train_mask],
                      node_interact_times=node_interact_times[train_mask],
                      edge_ids=edge_ids[train_mask], labels=labels[train_mask], snapshots=snapshots[train_mask])

    val_mask = np.logical_and(node_interact_times <= test_time, node_interact_times > val_time)
    test_mask = node_interact_times > test_time

    # validation and test data
    val_data = Data(src_node_ids=src_node_ids[val_mask], dst_node_ids=dst_node_ids[val_mask],
                    node_interact_times=node_interact_times[val_mask], edge_ids=edge_ids[val_mask], labels=labels[val_mask], snapshots=snapshots[val_mask])

    test_data = Data(src_node_ids=src_node_ids[test_mask], dst_node_ids=dst_node_ids[test_mask],
                     node_interact_times=node_interact_times[test_mask], edge_ids=edge_ids[test_mask], labels=labels[test_mask], snapshots=snapshots[test_mask])

    print("The dataset has {} interactions, involving {} different nodes".format(full_data.num_interactions, full_data.num_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
        train_data.num_interactions, train_data.num_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
        val_data.num_interactions, val_data.num_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
        test_data.num_interactions, test_data.num_unique_nodes))

    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, None, None, node_snap_counts


def load_dblp3_data():
    """
    RTGCN의 DBLP3 데이터를 DTFormer 형식으로 로드합니다.
    """
    import numpy as np
    from scipy.sparse import csr_matrix
    
    # DBLP3.npz 파일 로드
    data = np.load('RTGCN/data/DBLP3/DBLP3.npz')
    
    # 데이터 추출
    adjs = data['adjs']  # (10, 4257, 4257) - 10개의 타임스텝에 대한 인접 행렬
    attmats = data['attmats']  # (4257, 10, 100) - 노드 특성
    labels = data['labels']  # (4257, 3) - 노드 레이블
    
    num_nodes = adjs.shape[1]  # 4257
    num_timesteps = adjs.shape[0]  # 10
    
    # 노드 특성을 DTFormer 형식으로 변환
    node_features = np.zeros((num_nodes, 172))  # DTFormer는 172차원 특성을 사용
    node_features[:, :100] = attmats[:, -1, :]  # 마지막 타임스텝의 특성 사용
    
    # 엣지 정보 생성
    edge_list = []
    edge_times = []
    
    for t in range(num_timesteps):
        # 현재 타임스텝의 인접 행렬
        adj_matrix = adjs[t]
        # 엣지가 있는 위치 찾기
        src_nodes, dst_nodes = np.nonzero(adj_matrix)
        
        # 현재 타임스텝의 엣지 추가
        edge_list.extend(list(zip(src_nodes, dst_nodes)))
        edge_times.extend([t] * len(src_nodes))
    
    edge_list = np.array(edge_list)
    edge_times = np.array(edge_times)
    
    # 시간 정규화
    edge_times = (edge_times - edge_times.min()) / (edge_times.max() - edge_times.min())
    
    # 데이터 분할 (train: 0.7, val: 0.15, test: 0.15)
    num_edges = len(edge_times)
    indices = np.random.permutation(num_edges)
    train_size = int(0.7 * num_edges)
    val_size = int(0.15 * num_edges)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Data 클래스 인스턴스 생성
    train_data = Data(
        src_node_ids=edge_list[train_indices, 0],
        dst_node_ids=edge_list[train_indices, 1],
        node_interact_times=edge_times[train_indices],
        edge_ids=train_indices,
        labels=np.ones(len(train_indices)),  # 모든 엣지는 양성 샘플
        snapshots=np.ones(len(train_indices))  # 스냅샷 정보는 1로 설정
    )
    
    val_data = Data(
        src_node_ids=edge_list[val_indices, 0],
        dst_node_ids=edge_list[val_indices, 1],
        node_interact_times=edge_times[val_indices],
        edge_ids=val_indices,
        labels=np.ones(len(val_indices)),
        snapshots=np.ones(len(val_indices))
    )
    
    test_data = Data(
        src_node_ids=edge_list[test_indices, 0],
        dst_node_ids=edge_list[test_indices, 1],
        node_interact_times=edge_times[test_indices],
        edge_ids=test_indices,
        labels=np.ones(len(test_indices)),
        snapshots=np.ones(len(test_indices))
    )
    
    full_data = Data(
        src_node_ids=edge_list[:, 0],
        dst_node_ids=edge_list[:, 1],
        node_interact_times=edge_times,
        edge_ids=np.arange(len(edge_times)),
        labels=np.ones(len(edge_times)),
        snapshots=np.ones(len(edge_times))
    )
    
    return node_features, None, full_data, train_data, val_data, test_data, None, None, None
