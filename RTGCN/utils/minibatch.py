import random

import torch
import numpy as np
import scipy.sparse as sp
import torch_geometric as tg
from torch.utils.data import Dataset
from torch_geometric.data import Data

from utils.utilities import fixed_unigram_candidate_sampler, fixed_unigram_candidate_role_aware_sampler

class MyDataset(Dataset):
    def __init__(self, args, graphs, features, adjs,  context_pairs):
        """
        Initialize the dataset with graphs, features, adjacency matrices, and context pairs.
        """
        super(MyDataset, self).__init__()
        self.args = args
        self.graphs = graphs
        self.features = [self._preprocess_features(feat) for feat in features]
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]
        self.time_steps = args.time_steps
        self.context_pairs = context_pairs
        self.max_positive = args.neg_sample_size
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        self.degs = self.construct_degs()
        self.pyg_graphs = self._build_pyg_graphs()
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """
        Normalize the adjacency matrix using the GCN formula.
        """
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized

    def _preprocess_features(self, features):
        """
        Row-normalize feature matrix and convert to tuple representation
        """
        features = np.array(features)
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def construct_degs(self):
        """
        Compute node degrees in each graph snapshot
        """
        degs = []
        for i in range(self.min_t, self.time_steps):
            G = self.graphs[i]
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs

    def _build_pyg_graphs(self):
        """
        Construct PyTorch Geometric graph data objects from the adjacency matrices and features.
        This method converts each graph in the dataset to the PyTorch Geometric format, which is
        used for graph neural network models in PyTorch.
        """
        pyg_graphs = []
        for feat, adj in zip(self.features, self.adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        """
        Return the total number of nodes in the last snapshot.
        """
        return len(self.train_nodes)
    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def __getitem__(self, index):
        """
        Fetch the pre-computed data items for the node at the specified index.
        """
        node = self.train_nodes[index]
        return self.data_items[node]
    
    def __createitems__(self):
        """
        Prepares the dataset items for training by constructing positive and negative context pairs
        for each node over different time steps. It handles temporal relationships and negative sampling.
        """
        self.data_items = {}

        # Iterate over all nodes in the last snapshot of the graph series
        for node in list(self.graphs[self.time_steps-1].nodes()):
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []

            # Gather positive context pairs for each node over the specified time range
            for t in range(self.min_t, self.time_steps):
                node_1 = []
                node_2 = []
                if len(self.context_pairs[t][node]) > self.max_positive:
                    # Limit the number of positive samples to max_positive
                    node_1.extend([node]* self.max_positive)
                    node_2.extend(np.random.choice(self.context_pairs[t][node], self.max_positive, replace=False))
                else:
                    node_1.extend([node]* len(self.context_pairs[t][node]))
                    node_2.extend(self.context_pairs[t][node])

                # Ensure the node lists have the same length
                assert len(node_1) == len(node_2)
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            # Convert lists to PyTorch tensors
            # node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            # node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            node_1_list = [torch.tensor(node, dtype=torch.long) for node in node_1_all_time]
            node_2_list = [torch.tensor(node, dtype=torch.long) for node in node_2_all_time]

            node_2_negative = []
            # Generate negative samples for each time step
            for t in range(len(node_2_list)):
                degree = self.degs[t]
                node_positive = node_2_list[t][:, None]
                node_negative = fixed_unigram_candidate_sampler(true_clasees=node_positive,
                                                                num_true=1,
                                                                num_sampled=self.args.neg_sample_size,
                                                                unique=False,
                                                                distortion=0.75,
                                                                unigrams=degree)
                node_2_negative.append(node_negative)
            # node_2_neg_list = [torch.LongTensor(node) for node in node_2_negative]
            node_2_neg_list = [torch.tensor(node, dtype=torch.long) for node in node_2_negative]
            feed_dict['node_1']=node_1_list
            feed_dict['node_2']=node_2_list
            feed_dict['node_2_neg']=node_2_neg_list

            self.data_items[node] = feed_dict

    def num_training_batches(self):
        """
        Compute the number of training batches (using batch size)
        """
        return len(self.train_nodes) // self.batch_size + 1


    def shuffle(self):
        """
        Re-shuffle the training set.
        Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        """ Reset batch number"""
        self.train_nodes =  list(self.graphs[self.time_steps-1].nodes())
        self.batch_num = 0


    @staticmethod
    def collate_fn(samples):
        """
        Collate function to merge a list of samples into a batch for loading.
        """
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                concate.append(torch.cat([data[t] for data in data_list]))
            batch_dict[key] = concate
        return batch_dict



class MyDatasetRole(Dataset):
    def __init__(self, args, graphs, features, adjs,  context_pairs, train_role_graph):
        """
        Initialize the dataset with graphs, features, adjacency matrices, and context pairs.
        """
        super(MyDatasetRole, self).__init__()
        self.args = args
        self.graphs = graphs
        self.features = [self._preprocess_features(feat) for feat in features]
        self.adjs = [self._normalize_graph_gcn(a)  for a  in adjs]
        self.time_steps = args.time_steps
        self.context_pairs = context_pairs
        self.max_positive = args.neg_sample_size
        self.train_nodes = list(self.graphs[self.time_steps-1].nodes()) # all nodes in the graph.
        self.min_t = max(self.time_steps - self.args.window - 1, 0) if args.window > 0 else 0
        self.degs = self.construct_degs()
        self.pyg_graphs = self._build_pyg_graphs()
        self.train_role_graph = train_role_graph #### updated
        self.__createitems__()

    def _normalize_graph_gcn(self, adj):
        """
        Normalize the adjacency matrix using the GCN formula.
        """
        adj = sp.coo_matrix(adj, dtype=np.float32)
        adj_ = adj + sp.eye(adj.shape[0], dtype=np.float32)
        rowsum = np.array(adj_.sum(1), dtype=np.float32)
        degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten(), dtype=np.float32)
        adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        return adj_normalized

    def _preprocess_features(self, features):
        """
        Row-normalize feature matrix and convert to tuple representation
        """
        features = np.array(features)
        rowsum = np.array(features.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        features = r_mat_inv.dot(features)
        return features

    def construct_degs(self):
        """
        Compute node degrees in each graph snapshot
        """
        degs = []
        for i in range(self.min_t, self.time_steps):
            G = self.graphs[i]
            deg = []
            for nodeid in G.nodes():
                deg.append(G.degree(nodeid))
            degs.append(deg)
        return degs

    def _build_pyg_graphs(self):
        """
        Construct PyTorch Geometric graph data objects from the adjacency matrices and features.
        This method converts each graph in the dataset to the PyTorch Geometric format, which is
        used for graph neural network models in PyTorch.
        """
        pyg_graphs = []
        for feat, adj in zip(self.features, self.adjs):
            x = torch.Tensor(feat)
            edge_index, edge_weight = tg.utils.from_scipy_sparse_matrix(adj)
            data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
            pyg_graphs.append(data)
        return pyg_graphs

    def __len__(self):
        """
        Return the total number of nodes in the last snapshot.
        """
        return len(self.train_nodes)
    
    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def __getitem__(self, index):
        """
        Fetch the pre-computed data items for the node at the specified index.
        """
        node = self.train_nodes[index]
        return self.data_items[node]
    
    def __createitems__(self):
        """
        Prepares the dataset items for training by constructing positive and negative context pairs
        for each node over different time steps. It handles temporal relationships and negative sampling.
        """
        self.data_items = {}

        # Iterate over all nodes in the last snapshot of the graph series
        for node in list(self.graphs[self.time_steps-1].nodes()):
            feed_dict = {}
            node_1_all_time = []
            node_2_all_time = []

            # Gather positive context pairs for each node over the specified time range
            for t in range(self.min_t, self.time_steps):
                node_1 = []
                node_2 = []
                current_context_pairs = self.context_pairs[t].get(node, []) # Use .get for safety, default to empty list #### updated
                if len(current_context_pairs) > self.max_positive:
                    # Limit the number of positive samples to max_positive
                    node_1.extend([node]* self.max_positive)
                    node_2.extend(np.random.choice(current_context_pairs, self.max_positive, replace=False).tolist()) #### updated - Ensure list output
                else:
                    node_1.extend([node]* len(current_context_pairs))
                    node_2.extend(current_context_pairs)

                # Ensure the node lists have the same length
                assert len(node_1) == len(node_2) or len(node_1) == 0, f"Mismatch len(node_1)={len(node_1)} != len(node_2)={len(node_2)} for node {node} at time {t}" #### updated
                node_1_all_time.append(node_1)
                node_2_all_time.append(node_2)

            # Convert lists to PyTorch tensors
            # node_1_list = [torch.LongTensor(node) for node in node_1_all_time]
            # node_2_list = [torch.LongTensor(node) for node in node_2_all_time]
            # node_1_list = [torch.tensor(node, dtype=torch.long) for node in node_1_all_time]
            # node_2_list = [torch.tensor(node, dtype=torch.long) for node in node_2_all_time]

            node_2_negative = []
            for t_idx_in_window, t in enumerate(range(self.min_t, self.time_steps)): #### updated from now onwards
                positive_nodes_t_list = node_2_all_time[t_idx_in_window]

                # If there are no positive nodes for this anchor at this time step, no negatives are needed for L_c
                if not positive_nodes_t_list: 
                     node_2_negative.append([]) 
                     continue 

                # Get the unigrams (degrees) for the current time step (relative index in self.degs)
                degree_t = self.degs[t_idx_in_window] 

                # Get the structural role mapping for the current absolute time step 't'
                # self.train_role_graph is a list [roles_t0, roles_t1, ...]
                # roles_tt is a dict {role_id: [node_idx, ...]}
                role_mapping_t = self.train_role_graph[t] if self.train_role_graph is not None else {} 

                # Sample negatives using the updated sampler function
                # The sampler returns a flat list of negatives for all positive pairs of this anchor at time t
                neg_samples_flat = fixed_unigram_candidate_role_aware_sampler( 
                    anchor_node=node, 
                    positive_nodes_list=positive_nodes_t_list, 
                    num_sampled_per_pos=self.args.neg_sample_size,
                    unique=False, # Original code used False 
                    distortion=0.75, # Original code used 0.75, but sampler doesn't apply it directly 
                    unigrams=degree_t, # Degrees for weighted sampling
                    role_mapping_t=role_mapping_t # Role information for exclusion
                )
                node_2_negative.append(neg_samples_flat)

            # Convert lists of lists (node_1_all_time, node_2_all_time, node_2_negative)
            # into lists of tensors (one tensor per time step in the window)
            # Fix 2 applied here and below to potentially improve performance
            node_1_list_tensors = [torch.tensor(nodes, dtype=torch.long) for nodes in node_1_all_time] 
            node_2_list_tensors = [torch.tensor(nodes, dtype=torch.long) for nodes in node_2_all_time] 
            node_2_neg_list_tensors = [torch.tensor(nodes, dtype=torch.long) for nodes in node_2_negative] 

            # Store the processed data (list of tensors) for this anchor node
            feed_dict['node_1']=node_1_list_tensors
            feed_dict['node_2']=node_2_list_tensors 
            feed_dict['node_2_neg']=node_2_neg_list_tensors 

            self.data_items[node] = feed_dict

    def num_training_batches(self):
        """
        Compute the number of training batches (using batch size)
        """
        return len(self.train_nodes) // self.batch_size + 1


    def shuffle(self):
        """
        Re-shuffle the training set.
        Also reset the batch number.
        """
        self.train_nodes = np.random.permutation(self.train_nodes)
        self.batch_num = 0

    def test_reset(self):
        """ Reset batch number"""
        self.train_nodes =  list(self.graphs[self.time_steps-1].nodes())
        self.batch_num = 0


    @staticmethod
    def collate_fn(samples):
        """
        Collate function to merge a list of samples into a batch for loading.
        """
        batch_dict = {}
        for key in ["node_1", "node_2", "node_2_neg"]:
            data_list = []
            for sample in samples:
                data_list.append(sample[key])
            concate = []
            for t in range(len(data_list[0])):
                concate.append(torch.cat([data[t] for data in data_list]))
            batch_dict[key] = concate
        return batch_dict


    
