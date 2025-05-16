import random

import numpy as np
import copy
import networkx as nx
from collections import defaultdict
from utils.random_walk import Graph_RandomWalk, alias_setup, alias_draw

import torch



"""Random walk-based pair generation."""

def run_random_walks_n2v(graph, adj, num_walks, walk_len):
    """
    Perform random walks on the graph using node2vec's sampling strategy.

    Args:
        graph (networkx.Graph): The graph on which random walks are to be performed.
        adj (np.array): The adjacency matrix of the graph.
        num_walks (int): The number of walks to perform for each node.
        walk_len (int): The length of each random walk.

    Returns:
        dict: A dictionary where keys are start nodes and values are lists of context nodes found in random walks.
    """
    # Initialize a NetworkX graph from the given data
    nx_G = nx.Graph()
    for e in graph.edges():
        nx_G.add_edge(e[0], e[1])
    for edge in graph.edges():
        nx_G[edge[0]][edge[1]]['weight'] = adj[edge[0], edge[1]]

    # Set up the graph for node2vec processing
    G = Graph_RandomWalk(nx_G, False, 1.0, 1.0)
    G.preprocess_transition_probs()

    # Generate walks and collect context pairs within a defined window size
    walks = G.simulate_walks(num_walks, walk_len)
    WINDOW_SIZE = 10
    pairs = defaultdict(list)
    pairs_cnt = 0
    for walk in walks:
        for word_index, word in enumerate(walk):
            for nb_word in walk[max(word_index - WINDOW_SIZE, 0): min(word_index + WINDOW_SIZE, len(walk)) + 1]:
                if nb_word != word:
                    pairs[word].append(nb_word)
                    pairs_cnt += 1
    print("# nodes with random walk samples: {}".format(len(pairs)))
    print("# sampled pairs: {}".format(pairs_cnt))
    return pairs

# Helper function to get roles for a node from the role mapping dictionary
def get_node_roles_from_mapping(node_idx, role_mapping_t): 
    """Helper to find roles for a node from the role dictionary.""" 
    roles = set() 
    if isinstance(role_mapping_t, dict): 
        for role_id, node_list in role_mapping_t.items(): 
            if node_idx in node_list: 
                roles.add(role_id) 
    return roles 

def fixed_unigram_candidate_role_aware_sampler(anchor_node, positive_nodes_list, num_sampled_per_pos, unique, distortion, unigrams, role_mapping_t): 
    """
    Samples negative examples for the connective proximity loss. Samples `num_sampled_per_pos`
    negatives for *each* positive node associated with the anchor node (u),
    excluding the positive node itself and any node that shares a structural role
    with the anchor node (u) at the current time step t. Returns a flat list. #### updated

    Args:
        anchor_node (int): The index of the anchor node (u). #### updated
        positive_nodes_list (list): List of positive node indices (v) for the anchor at time t. #### updated
        num_sampled_per_pos (int): Number of negative classes to sample *per positive node*. #### updated
        unique (bool): If true, sample without replacement (per batch of negatives for a positive node). #### updated
        distortion (float): The distortion to apply to the unigram probabilities (not applied currently). #### updated
        unigrams (np.array): Unigram probability distribution (e.g., node degrees).
        role_mapping_t (dict): Dictionary mapping role IDs to lists of node indices for time t. #### updated

    Returns:
        list: Flat list of (len(positive_nodes_list) * num_sampled_per_pos) sampled negative node indices.
    """
    num_nodes = len(unigrams) #### updated
    all_sampled_negatives = [] #### updated

    # Identify roles of the anchor node (u)
    anchor_roles = get_node_roles_from_mapping(anchor_node, role_mapping_t) #### updated

    # Identify all nodes with the same role(s) as the anchor (u)
    same_role_nodes = set() #### updated
    if anchor_roles: #### updated
        for role_id in anchor_roles: #### updated
             if role_id in role_mapping_t: #### updated
                 same_role_nodes.update(role_mapping_t[role_id]) #### updated

    # Nodes to exclude from the sampling pool for *any* positive node associated with this anchor
    # Exclude the anchor node (u) itself and all nodes (including u) with the same role(s) as u.
    base_excluded_nodes = same_role_nodes.copy() #### updated
    base_excluded_nodes.add(anchor_node) #### updated

    # Create a mask for potentially valid candidates (excluding same-role nodes and anchor)
    initial_mask = np.ones(num_nodes, dtype=bool) #### updated
    # Ensure excluded nodes are valid indices before masking
    valid_base_excluded = [n for n in base_excluded_nodes if 0 <= n < num_nodes] #### updated
    initial_mask[valid_base_excluded] = False #### updated

    initial_candidate_pool = np.array(range(num_nodes))[initial_mask] #### updated
    initial_probs_raw = np.array(unigrams)[initial_mask] #### updated

    # Now, for each positive node v associated with the anchor u, sample negatives v'
    for positive_node_v in positive_nodes_list: #### updated
        # Create a mask specific to this positive node 'v'
        current_mask = initial_mask.copy() #### updated
        # Also exclude the positive node 'v' itself if it's a valid index
        if 0 <= positive_node_v < num_nodes: #### updated
             current_mask[positive_node_v] = False #### updated

        current_candidate_pool = np.array(range(num_nodes))[current_mask] #### updated
        current_probs_raw = np.array(unigrams)[current_mask] #### updated

        sum_current_probs = np.sum(current_probs_raw) #### updated

        sampled_negatives_for_pair = [] #### updated

        if len(current_candidate_pool) == 0: #### updated
             # No candidates available after exclusions
             if num_sampled_per_pos > 0: #### updated
                  # Append placeholders if needed to maintain structure/size expectations
                  # Returning anchor node itself as a dummy negative to prevent size mismatch.
                  sampled_negatives_for_pair = [anchor_node] * num_sampled_per_pos #### updated
        elif len(current_candidate_pool) < num_sampled_per_pos and unique: #### updated
              # Cannot sample enough unique negatives
              sampled_negatives_for_pair = current_candidate_pool.tolist() #### updated
              # Need to pad if not enough unique available and not unique is allowed
              if not unique and len(sampled_negatives_for_pair) < num_sampled_per_pos: #### updated
                  padding_needed = num_sampled_per_pos - len(sampled_negatives_for_pair) #### updated
                  # If sampling with replacement, use the current candidate pool and its probabilities
                  probs_to_sample_from = current_probs_raw / sum_current_probs if sum_current_probs > 0 else None #### updated
                  padding_samples = np.random.choice(current_candidate_pool, size=padding_needed, replace=True, p=probs_to_sample_from).tolist() #### updated
                  sampled_negatives_for_pair.extend(padding_samples) #### updated

        else:
            # Sample from the current filtered pool
            probs_to_sample_from = current_probs_raw / sum_current_probs if sum_current_probs > 0 else None #### updated
            sampled_indices_in_current = np.random.choice(len(current_candidate_pool), size=num_sampled_per_pos, replace=unique, p=probs_to_sample_from) #### updated
            sampled_negatives_for_pair = current_candidate_pool[sampled_indices_in_current].tolist() #### updated

        all_sampled_negatives.extend(sampled_negatives_for_pair) #### updated

    return all_sampled_negatives #### updated

def fixed_unigram_candidate_sampler(true_clasees, num_true, num_sampled, unique,  distortion, unigrams):
    """
    Samples negative examples using a unigram distribution with optional distortion.

    Args:
        true_classes (np.array): Array of true class indices.
        num_true (int): Number of true classes per example.
        num_sampled (int): Number of negative classes to sample.
        unique (bool): If true, sample without replacement.
        distortion (float): The distortion to apply to the unigram probabilities.
        unigrams (np.array): Unigram probability distribution.

    Returns:
        list: List of sampled negative examples for each input example.
    """
    assert true_clasees.shape[1] == num_true
    samples = []
    for i in range(true_clasees.shape[0]):
        dist = copy.deepcopy(unigrams)
        candidate = list(range(len(dist)))
        taboo = true_clasees[i].cpu().tolist()
        for tabo in sorted(taboo, reverse=True):
            candidate.remove(tabo)
            dist.pop(tabo)
        sample = np.random.choice(candidate, size=num_sampled, replace=unique, p=dist/np.sum(dist))
        samples.append(sample)
    return samples

def to_device(batch, device):
    """
    Transfers each tensor in the given batch to the specified device.

    Args:
        batch (dict): A dictionary containing lists of tensors.
        device (torch.device): The device to which the tensors should be moved.

    Returns:
        dict: A new dictionary with the same structure as 'batch', but with all tensors moved to the specified device.
    """
    feed_dict = copy.deepcopy(batch)

    # Extract tensor lists from the dictionary; assumes keys 'node_1', 'node_2', 'node_2_negative'
    node_1, node_2, node_2_negative = feed_dict.values()

    # Move each tensor list to the specified device
    feed_dict["node_1"] = [x.to(device) for x in node_1]
    feed_dict["node_2"] = [x.to(device) for x in node_2]
    feed_dict["node_2_neg"] = [x.to(device) for x in node_2_negative]

    # Return the updated dictionary with all tensors on the new device
    return feed_dict


        



