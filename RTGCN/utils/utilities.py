import numpy as np
import copy
# import networkx as nx
from collections import defaultdict
from utils.random_walk import Graph_RandomWalk 
import torch

def run_random_walks_n2v(graph_repr, adj_matrix_np, num_walks, walk_len, p=1.0, q=1.0):
    """
    graph_repr: dict, {'edge_index': torch.Tensor, 'num_nodes': int}
    adj_matrix_np: np.array, the raw adjacency matrix for the current graph snapshot.
                   Used to extract edge weights if the graph is weighted.
    num_walks: int
    walk_len: int
    p, q: float, Node2Vec parameters (default to 1.0 for simple weighted random walks)
    """
    edge_index = graph_repr['edge_index'] # This is a PyTorch tensor
    num_nodes = graph_repr['num_nodes']

    # Extract edge weights from adj_matrix_np for the given edge_index
    # edge_index is (2, num_edges). Rows are sources, cols are targets.
    # adj_matrix_np is (num_nodes, num_nodes)
    
    if edge_index.shape[1] == 0:
        print(f"# nodes with random walk samples: 0 (graph has no edges at this timestep)")
        print(f"# sampled pairs: 0")
        return defaultdict(list)

    # Convert edge_index to numpy if it's a tensor, for indexing adj_matrix_np
    if isinstance(edge_index, torch.Tensor):
        edge_index_np_for_weights = edge_index.cpu().numpy()
    else: 
        edge_index_np_for_weights = edge_index

    try:
        edge_weights = adj_matrix_np[edge_index_np_for_weights[0, :], edge_index_np_for_weights[1, :]]
        edge_weights = np.array(edge_weights).flatten()
    except IndexError as e:
        print(f"IndexError during weight extraction: {e}. Max index in edge_index: {edge_index_np_for_weights.max()}, adj_matrix_np shape: {adj_matrix_np.shape}")
        # Fallback to unweighted if indexing fails (e.g. num_nodes mismatch)
        edge_weights = np.ones(edge_index.shape[1], dtype=np.float32)


    G = Graph_RandomWalk(edge_index=edge_index, 
                         num_nodes=num_nodes,
                         is_directed=False, 
                         p=p,
                         q=q,
                         edge_weights=edge_weights) 

    G.preprocess_transition_probs() # NOTE(wsgwak): check its role in the code

    walks = G.simulate_walks(num_walks, walk_len)
    
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            start = max(word_index - WINDOW_SIZE, 0)
            end = min(word_index + WINDOW_SIZE + 1, len(walk))
            for i in range(start, end):
                if i == word_index:
                    continue
                nb_word = walk[i]
                pairs[word].append(nb_word)
                pairs_cnt += 1
                
    print(f"# nodes with random walk samples: {len(pairs)}")
    print(f"# sampled pairs: {pairs_cnt}")
    return pairs


def fixed_unigram_candidate_sampler(true_classes, num_true, num_sampled, unique, distortion, unigrams):
    """
    true_classes: torch.Tensor or np.array, shape (batch_size, num_true)
    unigrams: list, np.array, or torch.Tensor of node degrees or unigram probabilities.
    """
    if isinstance(true_classes, torch.Tensor):
        true_classes_np = true_classes.cpu().numpy()
    else:
        true_classes_np = np.array(true_classes)

    if not isinstance(unigrams, list): 
        if isinstance(unigrams, torch.Tensor):
            unigrams_list = unigrams.cpu().tolist()
        else: 
            unigrams_list = unigrams.tolist()
    else: 
        unigrams_list = list(unigrams) 

    assert true_classes_np.shape[1] == num_true, \
        f"true_classes.shape[1] {true_classes_np.shape[1]} != num_true {num_true}"
    
    samples = []
    if not unigrams_list: # Handle empty unigrams (e.g., graph with no nodes/edges)
        # print("Warning: Unigrams list is empty in fixed_unigram_candidate_sampler.")
        for _ in range(true_classes_np.shape[0]):
            samples.append(np.array([], dtype=int)) 
        return samples


    effective_distortion = distortion if distortion != 1.0 else None 
    
    try:
        unigrams_np = np.array(unigrams_list, dtype=float)
    except ValueError:
        # This can happen if unigrams_list contains non-numeric data (e.g. nested lists from bad degs)
        if unigrams_list:
            unigrams_np = np.ones(len(unigrams_list), dtype=float) / len(unigrams_list)
        else: 
             for _ in range(true_classes_np.shape[0]):
                samples.append(np.array([], dtype=int))
             return samples


    if effective_distortion:
        unigrams_distorted_np = np.power(unigrams_np, effective_distortion)
    else:
        unigrams_distorted_np = unigrams_np 

    if unigrams_distorted_np.size == 0: 
        sum_unigrams_distorted = 0
    else:
        sum_unigrams_distorted = np.sum(unigrams_distorted_np)

    if sum_unigrams_distorted == 0 and unigrams_distorted_np.size > 0 :
        probs_full = np.ones_like(unigrams_distorted_np) / max(1, len(unigrams_distorted_np)) # Avoid div by zero
    elif sum_unigrams_distorted == 0 and unigrams_distorted_np.size == 0:
        probs_full = np.array([]) 
    else:
        probs_full = unigrams_distorted_np / sum_unigrams_distorted
        
    full_candidate_list = list(range(len(unigrams_list)))

    for i in range(true_classes_np.shape[0]):
        taboo = true_classes_np[i, :].tolist()
        current_candidates = []
        current_probs = []
        
        if not full_candidate_list or probs_full.size == 0:
            samples.append(np.array([], dtype=int))
            continue

        for idx, prob_val in zip(full_candidate_list, probs_full):
            if idx not in taboo:
                current_candidates.append(idx)
                current_probs.append(prob_val)
        
        if not current_candidates: 
            samples.append(np.array([], dtype=int)) 
            continue
            
        current_probs_np = np.array(current_probs, dtype=float)
        sum_current_probs = np.sum(current_probs_np)

        if sum_current_probs == 0:
            if current_candidates:
                # print(f"Warning: Sum of current_probs is zero for sample {i}. Using uniform over remaining candidates.")
                final_probs = np.ones_like(current_probs_np) / len(current_probs_np)
            else:
                samples.append(np.array([], dtype=int))
                continue
        else:
            final_probs = current_probs_np / sum_current_probs
        
        actual_num_sampled = min(num_sampled, len(current_candidates))
        
        if actual_num_sampled == 0:
             samples.append(np.array([], dtype=int))
             continue

        try:
            sampled_indices = np.random.choice(
                current_candidates, 
                size=actual_num_sampled, 
                replace=not unique,
                p=final_probs
            )
            samples.append(sampled_indices)
        except ValueError as e:
            # This can happen if probabilities sum to non-1, or other issues with p
            try:
                sampled_indices = np.random.choice(
                    current_candidates,
                    size=actual_num_sampled,
                    replace=not unique
                )
                samples.append(sampled_indices)
            except Exception as fallback_e:
                # print(f"Fallback sampling also failed: {fallback_e}")
                samples.append(np.array([], dtype=int)) # Last resort

    return samples


def to_device(batch, device):
    feed_dict = copy.deepcopy(batch) # Deepcopy to avoid modifying original batch structure outside
    
    if "node_1" in feed_dict and isinstance(feed_dict["node_1"], list):
        feed_dict["node_1"] = [x.to(device) for x in feed_dict["node_1"]]
    
    if "node_2" in feed_dict and isinstance(feed_dict["node_2"], list):
        feed_dict["node_2"] = [x.to(device) for x in feed_dict["node_2"]]
        
    key_for_neg = "node_2_neg" if "node_2_neg" in feed_dict else "node_2_negative"

    if key_for_neg in feed_dict and isinstance(feed_dict[key_for_neg], list):
        feed_dict[key_for_neg] = [x.to(device) for x in feed_dict[key_for_neg]]
    
    return feed_dict