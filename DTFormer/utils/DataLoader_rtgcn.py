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
        self.unique_node_ids = set(src_node_ids) | set(dst_node_ids)
        self.num_unique_nodes = len(self.unique_node_ids)


def load_rtgcn_data_for_dtformer(dataset_path: str, val_ratio: float, test_ratio: float, num_snapshots: int):
    """
    RTGCN 데이터를 DTFormer 형식으로 변환하는 함수
    
    Args:
        dataset_path: RTGCN 데이터셋 경로
        val_ratio: 검증 데이터 비율
        test_ratio: 테스트 데이터 비율
        num_snapshots: 사용할 스냅샷 수
    
    Returns:
        node_raw_features: 노드 특성 행렬
        edge_raw_features: 엣지 특성 행렬
        full_data: 전체 데이터
        train_data: 학습 데이터
        val_data: 검증 데이터
        test_data: 테스트 데이터
        new_node_val_data: 새로운 노드 검증 데이터 (None)
        new_node_test_data: 새로운 노드 테스트 데이터 (None)
        node_snap_counts: 노드별 스냅샷 출현 횟수
    """
    # RTGCN 데이터 로드
    data = np.load(dataset_path, allow_pickle=True)
    adjs = data['adjs']  # (num_snapshots, num_nodes, num_nodes)
    attmats = data['attmats']  # (num_nodes, num_snapshots, feature_dim)
    
    num_nodes = adjs.shape[1]
    feature_dim = attmats.shape[2]
    
    # 노드 특성 변환
    node_raw_features = np.zeros((num_nodes + 1, NODE_FEAT_DIM))  # +1 for padding
    node_raw_features[1:, :feature_dim] = attmats[:, 0, :]  # 첫 번째 스냅샷의 특성 사용
    
    # 엣지 특성 생성 (RTGCN에는 엣지 특성이 없으므로 더미 데이터 생성)
    edge_raw_features = np.zeros((1, EDGE_FEAT_DIM))  # 더미 엣지 특성
    
    # 엣지 데이터 생성
    src_node_ids = []
    dst_node_ids = []
    node_interact_times = []
    edge_ids = []
    labels = []
    snapshots = []
    
    edge_idx = 0
    for snap_idx in range(min(num_snapshots, adjs.shape[0])):
        adj = adjs[snap_idx]
        src, dst = np.nonzero(adj)
        
        # 노드 ID가 1부터 시작하도록 조정
        src = src + 1
        dst = dst + 1
        
        src_node_ids.extend(src)
        dst_node_ids.extend(dst)
        node_interact_times.extend([snap_idx] * len(src))
        edge_ids.extend(range(edge_idx, edge_idx + len(src)))
        labels.extend([1] * len(src))
        snapshots.extend([snap_idx + 1] * len(src))
        
        edge_idx += len(src)
    
    # numpy 배열로 변환
    src_node_ids = np.array(src_node_ids, dtype=np.longlong)
    dst_node_ids = np.array(dst_node_ids, dtype=np.longlong)
    node_interact_times = np.array(node_interact_times, dtype=np.float64)
    edge_ids = np.array(edge_ids, dtype=np.longlong)
    labels = np.array(labels)
    snapshots = np.array(snapshots)
    
    # 노드별 스냅샷 출현 횟수 계산
    node_snap_counts = np.zeros((num_nodes + 1, num_snapshots))
    for snap_idx in range(min(num_snapshots, adjs.shape[0])):
        adj = adjs[snap_idx]
        node_counts = np.sum(adj, axis=1)
        node_snap_counts[1:, snap_idx] = node_counts
    
    # 패딩 노드의 스냅샷 카운트를 0으로 설정
    node_snap_counts[0, :] = 0
    
    # 데이터 분할
    total_interactions = len(node_interact_times)
    indices = np.arange(total_interactions)
    np.random.shuffle(indices)
    
    val_size = int(total_interactions * val_ratio)
    test_size = int(total_interactions * test_ratio)
    train_size = total_interactions - val_size - test_size
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # 데이터 객체 생성
    full_data = Data(src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                    node_interact_times=node_interact_times, edge_ids=edge_ids,
                    labels=labels, snapshots=snapshots)
    
    train_data = Data(src_node_ids=src_node_ids[train_indices], dst_node_ids=dst_node_ids[train_indices],
                     node_interact_times=node_interact_times[train_indices], edge_ids=edge_ids[train_indices],
                     labels=labels[train_indices], snapshots=snapshots[train_indices])
    
    val_data = Data(src_node_ids=src_node_ids[val_indices], dst_node_ids=dst_node_ids[val_indices],
                   node_interact_times=node_interact_times[val_indices], edge_ids=edge_ids[val_indices],
                   labels=labels[val_indices], snapshots=snapshots[val_indices])
    
    test_data = Data(src_node_ids=src_node_ids[test_indices], dst_node_ids=dst_node_ids[test_indices],
                    node_interact_times=node_interact_times[test_indices], edge_ids=edge_ids[test_indices],
                    labels=labels[test_indices], snapshots=snapshots[test_indices])
    
    return node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, None, None, node_snap_counts


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
        load_rtgcn_data_for_dtformer(data_file, val_r, test_r, num_snaps_to_use)

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
    
    