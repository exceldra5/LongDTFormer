import numpy as np
import random
import torch 

# Alias sampling functions remain the same as they operate on probabilities
def alias_setup(probs):
    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int_) # Use np.int_ for explicit integer type

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K * prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)

    while smaller and larger: # Use truthiness of lists
        small = smaller.pop()
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)
    return J, q

def alias_draw(J, q):
    K = len(J)
    kk = int(np.floor(np.random.rand() * K))
    if np.random.rand() < q[kk]:
        return kk
    else:
        return J[kk]

class Graph_RandomWalk():
    def __init__(self, edge_index, num_nodes, is_directed, p, q, edge_weights=None):
        """
        edge_index: torch.LongTensor of shape (2, num_edges) or NumPy array.
        num_nodes: int, total number of nodes in the graph.
        is_directed: bool.
        p, q: float, Node2Vec hyperparameters.
        edge_weights: torch.Tensor or NumPy array of shape (num_edges,), optional.
                      Corresponds to edges in edge_index.
        """
        self.is_directed = is_directed
        self.p = p
        self.q = q
        self.num_nodes = num_nodes

        if isinstance(edge_index, torch.Tensor):
            self.edge_index_np = edge_index.cpu().numpy()
        else:
            self.edge_index_np = np.array(edge_index, dtype=np.int_)
        
        if self.edge_index_np.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, num_edges)")

        if edge_weights is not None:
            if isinstance(edge_weights, torch.Tensor):
                self.edge_weights_np = edge_weights.cpu().numpy()
            else:
                self.edge_weights_np = np.array(edge_weights, dtype=np.float32)
            if self.edge_weights_np.shape[0] != self.edge_index_np.shape[1]:
                raise ValueError("edge_weights must have the same length as the number of edges in edge_index.")
        else:
            # Default to unweighted (weight 1 for all edges)
            self.edge_weights_np = np.ones(self.edge_index_np.shape[1], dtype=np.float32)

        # Build an adjacency list representation for quick neighbor lookup
        self.adj = [[] for _ in range(self.num_nodes)]
        self.edge_weights_map = {} # (u,v) -> weight

        for i in range(self.edge_index_np.shape[1]):
            u, v = self.edge_index_np[0, i], self.edge_index_np[1, i]
            weight = self.edge_weights_np[i]
            
            self.adj[u].append(v)
            self.edge_weights_map[(u,v)] = weight
            if not self.is_directed:
                self.adj[v].append(u)
                self.edge_weights_map[(v,u)] = weight 

        # Sort adjacency lists for consistent processing (important for alias table generation)
        for i in range(self.num_nodes):
            self.adj[i] = sorted(list(set(self.adj[i]))) 

        self.alias_nodes = {}
        self.alias_edges = {}


    def _get_neighbors(self, node_id):
        if 0 <= node_id < self.num_nodes:
            return self.adj[node_id]
        return []

    def _get_edge_weight(self, u, v):
        return self.edge_weights_map.get((u,v), 0) 

    def _has_edge(self, u, v):
        return (u,v) in self.edge_weights_map


    def node2vec_walk(self, walk_length, start_node):
        walk = [start_node]
        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = self._get_neighbors(cur) # Already sorted

            if cur_nbrs:
                if len(walk) == 1:
                    # First step: Use alias_nodes for transition probabilities from current node
                    walk.append(cur_nbrs[alias_draw(self.alias_nodes[cur][0], self.alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    # Subsequent steps: Use alias_edges for transition (prev, cur) -> next
                    alias_key = (prev, cur)
                    if alias_key in self.alias_edges:
                         next_node_idx_in_cur_nbrs = alias_draw(self.alias_edges[alias_key][0], self.alias_edges[alias_key][1])
                         walk.append(cur_nbrs[next_node_idx_in_cur_nbrs])
                    else:
                        break # Or sample uniformly from cur_nbrs
            else: 
                break
        return walk

    def simulate_walks(self, num_walks, walk_length):
        walks = []
        nodes = list(range(self.num_nodes))
        for _ in range(num_walks): 
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
        return walks

    def get_alias_edge(self, src, dst):
        unnormalized_probs = []
        dst_neighbors = self._get_neighbors(dst) # Already sorted

        for dst_nbr in dst_neighbors: 
            weight = self._get_edge_weight(dst, dst_nbr)
            if dst_nbr == src: 
                unnormalized_probs.append(weight / self.p)
            elif self._has_edge(dst_nbr, src): 
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight / self.q)
        
        norm_const = sum(unnormalized_probs)
        if norm_const > 0:
            normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
        else: # No valid transitions (e.g. all probs zero due to p/q or zero weights)
            if dst_neighbors:
                normalized_probs = [1.0 / len(dst_neighbors)] * len(dst_neighbors)
            else:
                normalized_probs = []


        if not normalized_probs and dst_neighbors: # Should not happen if norm_const > 0 path is taken correctly
            # print(f"Warning: get_alias_edge for ({src},{dst}) resulted in empty normalized_probs despite neighbors. Using uniform.")
            return alias_setup([1.0/len(dst_neighbors)]*len(dst_neighbors))
        elif not normalized_probs and not dst_neighbors:
             return alias_setup([])


        return alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        # Alias table for nodes (first step of a walk)
        for node in range(self.num_nodes):
            node_neighbors = self._get_neighbors(node) # Sorted
            if node_neighbors:
                unnormalized_probs = [self._get_edge_weight(node, nbr) for nbr in node_neighbors]
                norm_const = sum(unnormalized_probs)
                if norm_const > 0:
                    normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]
                else: 
                    normalized_probs = [1.0 / len(node_neighbors)] * len(node_neighbors) if node_neighbors else []
                self.alias_nodes[node] = alias_setup(normalized_probs)
            else: 
                self.alias_nodes[node] = alias_setup([])


        # Alias table for edges (subsequent steps of a walk)
        for u_node in range(self.num_nodes):
            v_node_neighbors = self._get_neighbors(u_node)
            for v_node in v_node_neighbors:
                self.alias_edges[(u_node, v_node)] = self.get_alias_edge(u_node, v_node)
                if not self.is_directed:
                    self.alias_edges[(v_node, u_node)] = self.get_alias_edge(v_node, u_node)
        return