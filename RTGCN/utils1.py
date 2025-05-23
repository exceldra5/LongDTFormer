import torch
import random
import os
import numpy as np
import torch.nn as nn
import math
# Removed: import scipy.sparse as sparse (will use torch.sparse)
import scipy # will be removed 
import statistics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score # precision_score removed as unused

def reset_parameters(named_parameters):
    for i in named_parameters():
        if len(i[1].size()) == 1:
            std = 1.0 / math.sqrt(i[1].size(0))
            nn.init.uniform_(i[1], -std, std)
        else:
            nn.init.xavier_normal_(i[1])

def set_random_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True) # PyTorch 1.8+ for reproducibility
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8' # NOTE(wsgwak): check
    
    torch.backends.cudnn.enabled = False 

class HyperG:
    def __init__(self, H, X=None, w=None):
        """
        H: PyTorch sparse_coo_tensor of shape (n_nodes, n_edges)
        X: PyTorch FloatTensor of shape (n_nodes, n_features), optional
        w: PyTorch FloatTensor of shape (n_edges,), optional
        """
        assert H.is_sparse, "H must be a PyTorch sparse tensor"
        assert H.ndim == 2

        self._H = H
        self._n_nodes = self._H.shape[0]
        self._n_edges = self._H.shape[1]
        self._device = H.device

        if X is not None:
            assert isinstance(X, torch.Tensor) and X.ndim == 2
            assert X.shape[0] == self._n_nodes
            self._X = X.to(self._device)
        else:
            self._X = None

        if w is not None:
            assert isinstance(w, torch.Tensor)
            self.w = w.reshape(-1).to(self._device)
            assert self.w.shape[0] == self._n_edges
        else:
            self.w = torch.ones(self._n_edges, device=self._device)

        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None

    def num_edges(self):
        return self._n_edges

    def num_nodes(self):
        return self._n_nodes

    def incident_matrix(self):
        return self._H

    def hyperedge_weights(self):
        return self.w

    def node_features(self):
        return self._X
    
    def _get_sparse_diag(self, diag_values, size_tuple):
        if not diag_values.is_sparse: # Ensure diag_values is dense for torch.diag
            diag_values = diag_values.to_dense()
        # Ensure diag_values is 1D for torch.diag
        if diag_values.ndim > 1:
            diag_values = diag_values.squeeze()
        
        # Check if diag_values is empty
        if diag_values.numel() == 0:
            return torch.sparse_coo_tensor(torch.empty((2,0), dtype=torch.long, device=self._device),
                                           torch.empty((0,), dtype=diag_values.dtype, device=self._device),
                                           size_tuple, device=self._device)

        # Create indices for diagonal elements
        indices = torch.arange(diag_values.shape[0], device=self._device).unsqueeze(0).repeat(2, 1)
        return torch.sparse_coo_tensor(indices, diag_values, size_tuple, device=self._device)


    def node_degrees(self):
        if self._DV is None:
            # H is (N, E), w is (E,). We want (N, N) diagonal matrix.
            # dv = sum_e H_ie * w_e for each node i
            
            # dv = torch.sparse.mm(self._H, self.w.reshape(-1, 1)).squeeze()
            # self._DV = self._get_sparse_diag(dv, (self._n_nodes, self._n_nodes))
            H_dense = self._H.to_dense()
            dv = torch.matmul(H_dense, self.w.reshape(-1, 1)).squeeze()
            self._DV = self._get_sparse_diag(dv, (self._n_nodes, self._n_nodes)) # This puts it back to sparse diag
            # If DV itself needs to be dense:
            # self._DV = torch.diag(dv)
        return self._DV

    def edge_degrees(self):
        if self._DE is None:
            # de = sum_n H_ne for each edge e
            # H is (N,E). H.T is (E,N). H.T.sum(dim=1) gives (E,)
            de = torch.sparse.sum(self._H, dim=0).to_dense() # to_dense for _get_sparse_diag
            self._DE = self._get_sparse_diag(de, (self._n_edges, self._n_edges))
        return self._DE

    def inv_edge_degrees(self):
        if self._INVDE is None:
            de = self.edge_degrees().coalesce().values() # Get diagonal values
            inv_de_val = torch.pow(de, -1.)
            inv_de_val[torch.isinf(inv_de_val) | torch.isnan(inv_de_val)] = 0 # Handle division by zero
            self._INVDE = self._get_sparse_diag(inv_de_val, (self._n_edges, self._n_edges))
        return self._INVDE

    def inv_square_node_degrees(self):
        if self._DV2 is None:
            dv = self.node_degrees().coalesce().values() # Get diagonal values
            dv2_val = torch.pow(dv + 1e-6, -0.5) # Add epsilon for stability
            dv2_val[torch.isinf(dv2_val) | torch.isnan(dv2_val)] = 0
            self._DV2 = self._get_sparse_diag(dv2_val, (self._n_nodes, self._n_nodes))
        return self._DV2

    def theta_matrix(self):
        if self._THETA is None:
            DV2 = self.inv_square_node_degrees()
            INVDE = self.inv_edge_degrees()
            # W_diag = self._get_sparse_diag(self.w, (self._n_edges, self._n_edges))
         
            W_invDE_diag_values = self.w * INVDE.coalesce().values()
            # W_invDE_diag = self._get_sparse_diag(W_invDE_diag_values, (self._n_edges, self._n_edges))


            DV2_dense = self.inv_square_node_degrees().to_dense()
            H_dense = self._H.to_dense()

            # W_invDE_diag_values was calculated earlier
            W_invDE_diag_dense = torch.diag(W_invDE_diag_values) # Creates a dense diagonal matrix

            # Theta_intermediate = (H_dense @ W_invDE_diag_dense) @ H_dense.T
            # Or step-by-step:
            Term_HWide_dense = torch.matmul(H_dense, W_invDE_diag_dense)
            Theta_intermediate_dense = torch.matmul(Term_HWide_dense, H_dense.T)

            self._THETA = torch.matmul(DV2_dense, torch.matmul(Theta_intermediate_dense, DV2_dense))

        return self._THETA

    def laplacian(self):
        if self._L is None:
            theta = self.theta_matrix() # This might be dense
            
            # Create sparse identity
            eye_indices = torch.arange(self._n_nodes, device=self._device).unsqueeze(0).repeat(2, 1)
            eye_values = torch.ones(self._n_nodes, device=self._device)
            I = torch.sparse_coo_tensor(eye_indices, eye_values, (self._n_nodes, self._n_nodes), device=self._device)
            
            if theta.is_sparse:
                self._L = I - theta
            else: # if theta is dense
                self._L = I.to_dense() - theta # Result will be dense
        return self._L

    def update_hyedge_weights(self, w):
        assert isinstance(w, torch.Tensor)
        self.w = w.reshape(-1).to(self._device)
        assert self.w.shape[0] == self._n_edges

        self._DV = None
        self._DV2 = None
        self._THETA = None
        self._L = None # Reset dependent attributes

    def update_incident_matrix(self, H):
        assert H.is_sparse and H.ndim == 2
        assert H.shape[0] == self._n_nodes
        assert H.shape[1] == self._n_edges
        
        self._H = H.to(self._device)
        self._DE = None
        self._DV = None
        self._INVDE = None
        self._DV2 = None
        self._THETA = None
        self._L = None


def gen_attribute_hg(n_nodes, role_dict2, Role_set, X=None, device='cpu'):
    if X is not None:
        assert isinstance(X, (np.ndarray, torch.Tensor))
        if isinstance(X, np.ndarray):
            X_pt = torch.from_numpy(X).float().to(device)
        else:
            X_pt = X.float().to(device)
        assert n_nodes == X_pt.shape[0]
    else:
        X_pt = None

    n_edges = len(Role_set)
    node_idx_list = []
    edge_idx_list = []

    for attr_name, nodes_in_attr in role_dict2.items():
        if attr_name in Role_set: # Ensure role is in the defined set
            nodes = sorted(list(map(int, nodes_in_attr))) # Ensure nodes are integers
            node_idx_list.extend(nodes)
            role_idx = Role_set.index(attr_name)
            edge_idx_list.extend([role_idx] * len(nodes))

    if not node_idx_list: # Handle case with no edges
        indices = torch.empty((2,0), dtype=torch.long, device=device)
        values = torch.empty((0,), dtype=torch.float, device=device)
    else:
        indices = torch.tensor([node_idx_list, edge_idx_list], dtype=torch.long, device=device)
        values = torch.ones(indices.shape[1], dtype=torch.float, device=device)

    H_sparse = torch.sparse_coo_tensor(indices, values, (n_nodes, n_edges), device=device)
    return HyperG(H_sparse, X=X_pt), H_sparse


def cross_role_hypergraphn_nodes(n_nodes, H_current_roles, role_dict1, role_dict2, Cross_role_Set, w_decay_factor, delta_t, X=None, device='cpu'):
    if X is not None:
        assert isinstance(X, (np.ndarray, torch.Tensor))
        if isinstance(X, np.ndarray):
            X_pt = torch.from_numpy(X).float().to(device)
        else:
            X_pt = X.float().to(device)
        assert n_nodes == X_pt.shape[0]
    else:
        X_pt = None
    
    assert H_current_roles.is_sparse, "H_current_roles must be a PyTorch sparse tensor"
    assert H_current_roles.shape[0] == n_nodes
    # n_current_role_edges = H_current_roles.shape[1] # This is for role_set, not cross_role_set

    n_cross_edges = len(Cross_role_Set)
    node_idx_list = []
    edge_idx_list = []
    values_list = [] # For weighted cross-role edges

    decay_val = torch.exp(torch.tensor(w_decay_factor * delta_t, dtype=torch.float, device=device)).item()

    for role_name_t2, nodes_t2 in role_dict2.items():
        if role_name_t2 in role_dict1 and role_name_t2 in Cross_role_Set:
            nodes_t1 = sorted(list(map(int, role_dict1[role_name_t2]))) # Nodes from t-1 having this role
            cross_role_idx = Cross_role_Set.index(role_name_t2)
            
            node_idx_list.extend(nodes_t1)
            edge_idx_list.extend([cross_role_idx] * len(nodes_t1))
            values_list.extend([decay_val] * len(nodes_t1))
            
    if not node_idx_list:
        H1_indices = torch.empty((2,0), dtype=torch.long, device=device)
        H1_values = torch.empty((0,), dtype=torch.float, device=device)
    else:
        H1_indices = torch.tensor([node_idx_list, edge_idx_list], dtype=torch.long, device=device)
        H1_values = torch.tensor(values_list, dtype=torch.float, device=device)

    H1_cross_temporal_sparse = torch.sparse_coo_tensor(H1_indices, H1_values, (n_nodes, n_cross_edges), device=device)
    
    if H_current_roles.shape[1] == H1_cross_temporal_sparse.shape[1]:
         H_combined = H_current_roles + H1_cross_temporal_sparse
    else:
        print(f"Warning: Dimensions for H_current_roles ({H_current_roles.shape}) and H1_cross_temporal_sparse ({H1_cross_temporal_sparse.shape}) differ in columns. Combination might be incorrect.")
        H_combined = H_current_roles 
        if H_current_roles.shape[1] != n_cross_edges:
             raise ValueError(f"H_current_roles.shape[1] ({H_current_roles.shape[1]}) must match n_cross_edges ({n_cross_edges}) for addition.")
        H_combined = H_current_roles + H1_cross_temporal_sparse


    return HyperG(H1_cross_temporal_sparse, X=X_pt), H_combined


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    if not isinstance(sparse_mx, scipy.sparse.coo_matrix):
        sparse_mx = sparse_mx.tocoo()
    sparse_mx = sparse_mx.astype(np.float32)
    
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def adj_to_edge(adjmatrix):
    # Converts a NumPy dense adjacency matrix to edge_index and edge_values (for unweighted graph)
    if isinstance(adjmatrix, torch.Tensor):
        adjmatrix_np = adjmatrix.cpu().numpy()
    else:
        adjmatrix_np = np.array(adjmatrix) 
    
    rows, cols = np.where(adjmatrix_np > 0) 
    
    # Create edge_index
    i = torch.from_numpy(np.vstack((rows, cols))).long()
    
    # Create edge_values (assuming 1 for existing edges, consistent with original LongTensor(values))
    v = torch.ones(i.shape[1], dtype=torch.long) 

    return [i, v]


def hypergraph_role_set(train_role_graph, time_step):
    role_set_all_ts = []
    for i in range(time_step):
        if i in train_role_graph:
            role_set_all_ts.extend(list(train_role_graph[i].keys()))
    
    Unique_Role_set = sorted(list(set(role_set_all_ts))) # Sort for consistent indexing

    Cross_role_Set = Unique_Role_set 
    # If it's roles present from t=1 onwards:
    # cross_role_temp = []
    # for j in range(1, time_step):
    #     if j in train_role_graph:
    #         cross_role_temp.extend(list(train_role_graph[j].keys()))
    # Cross_role_Set = sorted(list(set(cross_role_temp)))
    
    return Unique_Role_set, Cross_role_Set


def evaluate_node_classification(emb, labels_one_hot, datas_splits, device='cpu'):
    # emb: node embeddings from model (Torch tensor)
    # labels_one_hot: (Torch tensor or numpy array)
    # datas_splits: list of [train_idx, val_idx, test_idx] lists/numpy arrays
    
    if isinstance(labels_one_hot, torch.Tensor):
        labels_one_hot = labels_one_hot.cpu().numpy()
    labels = np.argmax(labels_one_hot, 1)

    if isinstance(emb, torch.Tensor):
        emb_np = emb.detach().cpu().numpy()
    else:
        emb_np = emb

    average_accs = []
    Temp_accs_all_ratios = []
    average_aucs = []
    Temp_aucs_all_ratios = []
    
    for train_nodes_over_ts in datas_splits: # Iterate over ratios
        temp_accs_current_ratio = []
        temp_aucs_current_ratio = []
        for t, split_indices in enumerate(train_nodes_over_ts): # Iterate over time steps for this ratio
            train_idx, val_idx, test_idx = split_indices[0], split_indices[1], split_indices[2]
            
            # Ensure indices are numpy arrays for indexing
            train_idx = np.array(train_idx, dtype=int)
            val_idx = np.array(val_idx, dtype=int)
            test_idx = np.array(test_idx, dtype=int)

            train_vec = emb_np[train_idx]
            train_y = labels[train_idx]

            # val_vec = emb_np[val_idx] # Not used in this version of evaluate
            # val_y = labels[val_idx]

            test_vec = emb_np[test_idx]
            test_y = labels[test_idx]

            # Filter out any empty arrays resulting from splits
            if train_vec.shape[0] == 0 or test_vec.shape[0] == 0 :
                print(f"Warning: Skipping a split due to empty train/test set. Train size: {train_vec.shape[0]}, Test size: {test_vec.shape[0]}")
                temp_accs_current_ratio.append(0.0) # Or handle as NaN or skip
                temp_aucs_current_ratio.append(0.0)
                continue

            clf = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=4000, random_state=42)
            clf.fit(train_vec, train_y)

            y_pred = clf.predict(test_vec)
            acc = accuracy_score(test_y, y_pred)
            
            if len(np.unique(test_y)) > 1 : # ROC AUC requires more than 1 class in y_true
                test_predict_proba = clf.predict_proba(test_vec)
                try:
                    # Check if all classes in training are present in test for 'ovr'
                    present_classes_train = np.unique(train_y)
                    present_classes_test = np.unique(test_y)
                    if not np.array_equal(np.sort(present_classes_train), np.sort(present_classes_test)) and len(present_classes_train) > len(present_classes_test):
                        # If test set is missing classes present in train, predict_proba might have different shape than expected for all classes
                         num_classes = labels_one_hot.shape[1] 
                         auc_labels = np.arange(num_classes)
                         test_roc_score = roc_auc_score(test_y, test_predict_proba, multi_class='ovr', average='macro', labels=auc_labels)

                    else:
                         test_roc_score = roc_auc_score(test_y, test_predict_proba, multi_class='ovr', average='macro')

                except ValueError as e:
                    print(f"ROC AUC calculation error: {e}. Test_y unique: {np.unique(test_y)}. Setting AUC to 0.0 for this split.")
                    test_roc_score = 0.0 # Handle cases where AUC cannot be computed
            else:
                test_roc_score = 0.0 # Or 0.5 if interpreted as random guessing, but 0.0 for problematic cases.

            temp_aucs_current_ratio.append(test_roc_score)
            temp_accs_current_ratio.append(acc)
            
        if temp_accs_current_ratio: # Avoid division by zero if all splits were empty
            average_accs.append(statistics.mean(temp_accs_current_ratio))
            average_aucs.append(statistics.mean(temp_aucs_current_ratio))
        else:
            average_accs.append(0.0)
            average_aucs.append(0.0)
            
        Temp_accs_all_ratios.append(temp_accs_current_ratio)
        Temp_aucs_all_ratios.append(temp_aucs_current_ratio)
        
    return average_accs, Temp_accs_all_ratios, average_aucs, Temp_aucs_all_ratios