import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch_geometric.utils as tg_utils
import numpy as np
# import scipy.sparse as sp # DO NOT USE IT!!! This is non deterministic

# Assuming fixed_unigram_candidate_sampler is correctly refactored in utilities
from utils.utilities import fixed_unigram_candidate_sampler


class MyDataset(Dataset):
    def __init__(self, args, graphs_repr_list, raw_features_list, raw_adjs_np_list, context_pairs):
        """
        args: Experiment arguments.
        graphs_repr_list: List of dicts, [{'edge_index': tensor, 'num_nodes': int}, ...].
        raw_features_list: List of NumPy arrays (node features per snapshot).
        raw_adjs_np_list: List of NumPy adjacency matrices (for normalization & weights).
        context_pairs: List of defaultdicts (context pairs per snapshot).
        """
        super(MyDataset, self).__init__()
        self.args = args
        self.graphs_repr = graphs_repr_list # Store for reference, e.g., num_nodes
        self.device = args.device if hasattr(args, 'device') else \
                      (torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.features = [self._preprocess_features(feat_np) for feat_np in raw_features_list]

        # Normalize adjacency matrices (graphs) using GCN formula
        self.normalized_graphs_pyg = []
        for i in range(len(raw_adjs_np_list)):
            adj_np = raw_adjs_np_list[i]
            num_nodes = self.graphs_repr[i]['num_nodes'] # Or adj_np.shape[0]
            
            # Convert adj_np to edge_index and edge_weights
            # Assuming adj_np can contain weights other than 0/1
            rows, cols = np.where(adj_np > 0) # Or a threshold if weights are continuous
            edge_index_unnormalized = torch.from_numpy(np.vstack((rows, cols))).long().to(self.device)
            edge_weights_unnormalized = torch.from_numpy(adj_np[rows, cols]).float().to(self.device)

            norm_edge_index, norm_edge_weight = self._normalize_graph_gcn(
                edge_index_unnormalized,
                num_nodes,
                edge_weights=edge_weights_unnormalized,
                dtype=torch.float32,
                device=self.device
            )
            self.normalized_graphs_pyg.append(
                Data(
                    x=self.features[i], 
                    edge_index=norm_edge_index, 
                    edge_attr=norm_edge_weight
                )
            )


        self.time_steps = args.time_steps
        self.context_pairs = context_pairs
        self.max_positive = getattr(args, 'max_positive', args.neg_sample_size) 
        
        if self.graphs_repr:
             self.train_nodes = list(range(self.graphs_repr[self.time_steps-1]['num_nodes']))
        else:
             self.train_nodes = []

        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        
        self.degs = self.construct_degs() 
        # self.pyg_graphs = self._build_pyg_graphs() # This is now self.normalized_graphs_pyg
        
        self.__createitems__()

    def _normalize_graph_gcn(self, edge_index, num_nodes, edge_weights=None, dtype=torch.float32, device='cpu'):
        # Add self-loops. If no weights, add_self_loops creates weights of 1.
        # If weights exist, it can fill with a value (e.g., 1) for self-loops.
        filled_value = 1.0
        edge_index_sl, edge_weights_sl = tg_utils.add_self_loops(
            edge_index, edge_weights, fill_value=filled_value, num_nodes=num_nodes
        )
        
        row, col = edge_index_sl[0], edge_index_sl[1]
        deg = tg_utils.degree(col, num_nodes=num_nodes, dtype=dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0 # Handle isolated nodes

        normalized_edge_weights = deg_inv_sqrt[row] * edge_weights_sl * deg_inv_sqrt[col]
        
        return edge_index_sl.to(device), normalized_edge_weights.to(device)

    def _preprocess_features(self, features_np):
        features = torch.from_numpy(np.array(features_np)).float().to(self.device)
        
        # Row-normalize feature matrix
        row_sum = features.sum(dim=1, keepdim=True)
        r_inv = torch.pow(row_sum, -1)
        r_inv[torch.isinf(r_inv)] = 0.
        features = features * r_inv 
        return features

    def construct_degs(self):
        # Compute node degrees in each graph snapshot for negative sampling
        degs_all_timesteps = []
        for t in range(self.min_t, self.time_steps):
            edge_idx_t = self.graphs_repr[t]['edge_index'].to(self.device) # ensure on correct device
            num_nodes_t = self.graphs_repr[t]['num_nodes']

            current_degs = tg_utils.degree(edge_idx_t[1], num_nodes=num_nodes_t, dtype=torch.float)
            
            # The sampler expects a list or np.array of degrees.
            degs_all_timesteps.append(current_degs.cpu().tolist())
        return degs_all_timesteps

    def __len__(self):
        return len(self.train_nodes)

    # def end(self): 
    #     return self.batch_num * self.batch_size >= len(self.train_nodes)

    def __getitem__(self, index):
        # train_nodes are indices 0 to N-1. 'index' is one such node_id.
        node_id = self.train_nodes[index] 
        return self.data_items[node_id] # data_items are pre-computed per node

    def __createitems__(self):
        """
        Prepares the dataset items for training by constructing positive and negative context pairs
        for each node over different time steps. It handles temporal relationships and negative sampling.
        """
        self.data_items = {}
        
        # Iterate over all nodes relevant for training (from the last snapshot)
        for node_id in self.train_nodes: # node_id is an integer from 0 to num_nodes-1
            feed_dict_for_node = {}
            node_1_all_time = [] # Source node (current node_id)
            node_2_all_time = [] # Positive context C_p(node_id)

            for t in range(self.min_t, self.time_steps):
                # context_pairs[t] is a defaultdict for snapshot t
                # context_pairs[t][node_id] is a list of context nodes for node_id at time t
                
                node_1_current_t = []
                node_2_current_t = []
                
                # Check if node_id is a key in context_pairs for this timestep
                if node_id not in self.context_pairs[t] or not self.context_pairs[t][node_id]:
                    node_1_all_time.append(torch.LongTensor([]).to(self.device))
                    node_2_all_time.append(torch.LongTensor([]).to(self.device))
                    continue # Move to next timestep

                # Actual positive context nodes for (node_id, t)
                positive_samples_for_node_t = self.context_pairs[t][node_id]

                if len(positive_samples_for_node_t) > self.max_positive:
                    node_1_current_t.extend([node_id] * self.max_positive)
                    chosen_pos_samples = np.random.choice(
                        positive_samples_for_node_t, self.max_positive, replace=False
                    )
                    node_2_current_t.extend(chosen_pos_samples)
                else:
                    node_1_current_t.extend([node_id] * len(positive_samples_for_node_t))
                    node_2_current_t.extend(positive_samples_for_node_t)
                
                node_1_all_time.append(torch.LongTensor(node_1_current_t).to(self.device))
                node_2_all_time.append(torch.LongTensor(node_2_current_t).to(self.device))

            # feed_dict_for_node['node_1'] = node_1_all_time # List of Tensors
            # feed_dict_for_node['node_2'] = node_2_all_time # List of Tensors

            # Generate negative samples for each time step
            node_2_neg_all_time = []
            for t_idx in range(len(node_2_all_time)): 
                if node_2_all_time[t_idx].numel() == 0: 
                    node_2_neg_all_time.append(torch.LongTensor([]).to(self.device))
                    continue

                # true_classes for sampler: (num_positive_samples_at_t, 1)
                # node_2_all_time[t_idx] is (num_positive_samples_at_t,)
                node_positive_for_sampler = node_2_all_time[t_idx].unsqueeze(1) 

                degree_dist_for_t = self.degs[t_idx]
                
                # Check if degree_dist_for_t is empty (e.g. graph had no nodes/edges)
                if not degree_dist_for_t:
                    # print(f"Warning: Degree distribution empty for timestep index {t_idx} for node {node_id}. Skipping negative sampling.")
                    node_2_neg_all_time.append(torch.LongTensor([]).to(self.device)) # No negatives if no degree dist
                    continue


                node_negative_np = fixed_unigram_candidate_sampler(
                    true_classes=node_positive_for_sampler.cpu(), # Sampler expects CPU tensor or np array
                    num_true=1,
                    num_sampled=self.args.neg_sample_size,
                    unique=False, 
                    distortion=0.75, # Standard value
                    unigrams=degree_dist_for_t
                ) 
                
                if node_negative_np and isinstance(node_negative_np, list) and \
                   all(isinstance(arr, np.ndarray) for arr in node_negative_np):
                    if all(arr.ndim > 0 for arr in node_negative_np if arr.size >0): # ensure not list of empty arrays
                         node_negative_tensor = torch.from_numpy(np.stack(node_negative_np)).long().to(self.device)
                    else: 
                        num_pos_samples = node_positive_for_sampler.shape[0]
                        node_negative_tensor = torch.LongTensor(num_pos_samples, self.args.neg_sample_size).fill_(0).to(self.device) 
                        # print(f"Warning: Could not stack negative samples for node {node_id}, t_idx {t_idx}. Using zeros.")

                else: 
                    num_pos_samples = node_positive_for_sampler.shape[0]
                    node_negative_tensor = torch.LongTensor(num_pos_samples, self.args.neg_sample_size).fill_(0).to(self.device) # Placeholder
                    # print(f"Warning: Negative sampler returned unexpected format for node {node_id}, t_idx {t_idx}. Using zeros.")

                node_2_neg_all_time.append(node_negative_tensor)
            
            feed_dict_for_node['node_1'] = node_1_all_time
            feed_dict_for_node['node_2'] = node_2_all_time
            feed_dict_for_node['node_2_neg'] = node_2_neg_all_time
            self.data_items[node_id] = feed_dict_for_node

    @staticmethod
    def collate_fn(samples):
        # Samples is a list of feed_dict_for_node from __getitem__
        batch_dict = {}
        if not samples: return batch_dict 

        # Keys are 'node_1', 'node_2', 'node_2_neg'
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list_for_key = [sample[key] for sample in samples if key in sample]
            
            if not data_list_for_key : continue 

            concatenated_timesteps = []
            num_timesteps_in_batch = len(data_list_for_key[0]) 

            for t in range(num_timesteps_in_batch):
                # Collect all tensors for current timestep 't' from all samples in the batch
                tensors_for_current_t = [sample_data_for_key[t] for sample_data_for_key in data_list_for_key if t < len(sample_data_for_key)]
                
                non_empty_tensors_for_t = [tensor for tensor in tensors_for_current_t if tensor.numel() > 0]

                if non_empty_tensors_for_t:
                    concatenated_timesteps.append(torch.cat(non_empty_tensors_for_t))
                else:
                    device = samples[0][key][0].device if samples and samples[0][key] and samples[0][key][0].numel() > 0 else \
                             (torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                    if key == "node_2_neg":
                        concatenated_timesteps.append(torch.LongTensor([]).to(device)) # Simplistic empty
                    else:
                        concatenated_timesteps.append(torch.LongTensor([]).to(device))
            
            batch_dict[key] = concatenated_timesteps
        return batch_dict