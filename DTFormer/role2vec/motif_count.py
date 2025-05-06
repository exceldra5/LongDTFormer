"""Heterogeneous Motif Counting Tool."""

import math
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from sklearn.decomposition import NMF
from networkx.generators.atlas import *
from joblib import Parallel, delayed
import os


def _process_node_list_for_motif_counts(nodes_list_item, main_graph, size, interesting_graphs, categories, unique_motif_count):
    """
    Helper function to process a single list of nodes (a potential graphlet)
    and calculate its contribution to the orbit counts.
    This function runs in a separate process.
    """
    sub_gr = main_graph.subgraph(nodes_list_item)
    graphs = interesting_graphs[size]

    # Local dictionary to store counts for nodes in this subgraph
    # Initialize only for the nodes present in this subgraph
    local_features_contribution = {node: {i: 0 for i in range(unique_motif_count)} for node in sub_gr.nodes()}

    # Find which benchmark graphlet structure this subgraph is isomorphic to
    for index, graph in enumerate(graphs):
        if nx.is_isomorphic(sub_gr, graph):
            # Found a match, calculate contributions for nodes in this subgraph
            for node in sub_gr.nodes():
                # Get the degree within the subgraph
                node_degree_in_subgraph = sub_gr.degree(node)
                # Check if the degree exists in the categories for this graphlet structure
                if node_degree_in_subgraph in categories[size][index]:
                    # Get the unique orbit ID
                    orbit_id = categories[size][index][node_degree_in_subgraph]
                    # Increment the count for this node and orbit
                    local_features_contribution[node][orbit_id] += 1
                # else:
                #     print(f"Warning: Node {node} has degree {node_degree_in_subgraph} in subgraph of size {size}, but this degree is not in categories[{size}][{index}]. Skipping count.")
            # Once isomorphic, no need to check other benchmark graphs of the same size
            return local_features_contribution # Return the local counts

    # If no isomorphic match is found for any benchmark graphlet of this size
    # (This shouldn't typically happen if node_lists contains connected components matching benchmark sizes)
    # print(f"Warning: No isomorphic benchmark graph found for nodes {nodes_list_item} of size {size}.")
    return {} # Return an empty dict if no contribution


class MotifCounterMachine(object):
    """
    Motif and Orbit Counting Tool.
    """
    def __init__(self, graph, args):
        """
        Initializing the object.
        :param graph: NetworkX graph.
        :param args: Arguments object.
        """
        self.graph = graph
        self.args = args
        self.n_jobs = os.cpu_count() // 2 - 4 if os.cpu_count() > 1 else 1 # Use all but one core, or 1 if only one

    def create_edge_subsets(self):
        """
        Collecting nodes that form graphlets.
        """
        self.edge_subsets = dict()
        subsets = [[edge[0], edge[1]] for edge in self.graph.edges()]
        self.edge_subsets[2] = subsets
        unique_subsets = dict()
        for i in range(3, self.args.graphlet_size+1):
            for subset in tqdm(subsets):
                for node in subset:
                    for neb in self.graph.neighbors(node):
                        new_subset = subset+[neb]
                        if len(set(new_subset)) == i:
                            new_subset.sort()
                            unique_subsets[tuple(new_subset)] = 1
            subsets = [list(k) for k, v in unique_subsets.items()]
            self.edge_subsets[i] = subsets
            unique_subsets = dict()

    def enumerate_graphs(self):
        """
        Enumerating connected benchmark graphlets.
        """
        graphs = graph_atlas_g()
        self.interesting_graphs = {i: [] for i in range(2, self.args.graphlet_size+1)}
        for graph in graphs:
            if graph.number_of_nodes() > 1 and  graph.number_of_nodes() < self.args.graphlet_size+1:
                if nx.is_connected(graph):
                    self.interesting_graphs[graph.number_of_nodes()].append(graph)

    def enumerate_categories(self):
        """
        Enumerating orbits in graphlets.
        """
        main_index = 0
        self.categories = dict()
        for size, graphs in self.interesting_graphs.items():
            self.categories[size] = dict()
            for index, graph in enumerate(graphs):
                self.categories[size][index] = dict()
                degrees = list(set([graph.degree(node) for node in graph.nodes()]))
                for degree in degrees:
                    self.categories[size][index][degree] = main_index
                    main_index = main_index + 1
        self.unique_motif_count = main_index + 1

    # def setup_features(self):
    #     """
    #     Calculating the graphlet orbit counts.
    #     """
    #     self.features = {n: {i: 0 for i in range(self.unique_motif_count)} for n in self.graph.nodes()}
    #     for size, node_lists in self.edge_subsets.items():
    #         # print("size", size)
    #         # print("node_lists", len(node_lists))
    #         # continue
    #         graphs = self.interesting_graphs[size]
    #         for nodes in tqdm(node_lists):
    #             sub_gr = self.graph.subgraph(nodes)
    #             for index, graph in enumerate(graphs):
    #                 if nx.is_isomorphic(sub_gr, graph):
    #                     for node in sub_gr.nodes():
    #                         self.features[node][self.categories[size][index][sub_gr.degree(node)]] += 1
    #                     break
    #     exit(0)
    
    def setup_features(self):
        """
        Calculating the graphlet orbit counts using parallel processing.
        """
        # Initialize features dictionary for all nodes with 0 counts
        # Use integer keys as self.features is populated from NetworkX node IDs
        self.features = {n: {i: 0 for i in range(self.unique_motif_count)} for n in self.graph.nodes()}

        print("\nCalculating graphlet orbit counts...")

        # Iterate through graphlet sizes
        for size, node_lists in self.edge_subsets.items():
            print(f"Processing graphlets of size {size} ({len(node_lists)} unique instances)...")

            if not node_lists:
                print(f"No unique graphlets of size {size} found. Skipping.")
                continue

            # Parallelize the processing of node_lists
            # Each item in node_lists (a list of nodes) is processed by _process_node_list_for_motif_counts
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=5)(
                delayed(_process_node_list_for_motif_counts)(
                    nodes_list_item,
                    self.graph, # Pass the main graph
                    size,
                    self.interesting_graphs,
                    self.categories,
                    self.unique_motif_count
                ) for nodes_list_item in tqdm(node_lists, desc=f"Size {size} Graphlets") # tqdm for outer loop progress
            )

            # Aggregate results from parallel processes
            print(f"Aggregating results for size {size}...")
            for local_contribution_dict in tqdm(results, desc=f"Aggregating Size {size}"):
                # local_contribution_dict is {node_id: {orbit_id: count}} for nodes in that graphlet
                for node_id, orbit_counts in local_contribution_dict.items():
                    for orbit_id, count in orbit_counts.items():
                         # Ensure orbit_id is within bounds (should be, but safety check)
                         if 0 <= orbit_id < self.unique_motif_count:
                            self.features[node_id][orbit_id] += count
                         else:
                             print(f"Warning: Received unexpected orbit_id {orbit_id} during aggregation.")


        print("Graphlet orbit counting complete.")

    # def create_tabular_motifs(self):
    #     """
    #     Creating tabular motifs for factorization.
    #     """
    #     self.binned_features = {node: [] for node in self.graph.nodes()}
    #     self.motifs = [[node]+[self.features[node][index] for index in  range(self.unique_motif_count )] for node in self.graph.nodes()]
    #     self.motifs = pd.DataFrame(self.motifs)
    #     self.motifs.columns = ["id"] + ["role_"+str(index) for index in range(self.unique_motif_count)]
    #     for index in range(self.unique_motif_count):
    #         features = self.motifs["role_"+str(index)].values.tolist()
    #         if sum(features) > 0:
    #             features = [math.log(feature+1) for feature in features]
    #             features = pd.qcut(features, self.args.quantiles, duplicates="drop", labels=False)
    #             for node in self.graph.nodes():
    #                 self.binned_features[node].append(str(int(index*self.args.quantiles + features[node])))

    # def join_strings(self):
    #     """
    #     Creating string labels by joining the individual quantile labels.
    #     """
    #     return {str(node): ["_".join(self.binned_features[node])] for node in self.graph.nodes()}

    def create_tabular_motifs(self):
        """
        Creating tabular motifs by binning orbit counts.
        Correctly handles node IDs vs. internal list indices.
        """
        # Get the list of nodes in a consistent order for indexing the binned features
        # We need this consistent order because pd.qcut outputs results indexed 0 to num_nodes-1
        node_list = list(self.graph.nodes())
        num_nodes = len(node_list)
        print("num_nodes", num_nodes)

        # Initialize binned_features using the original node IDs as keys
        self.binned_features = {str(node): [] for node in node_list}

        # Prepare motif counts in a DataFrame.
        # The order of nodes in this DataFrame will correspond to node_list.
        motif_data_for_df = []
        for node in node_list:
            # Ensure node is string key for self.features lookup
            node_str = str(node)
            # Ensure all expected feature indices exist for this node, default to 0 if not
            row = [node] + [self.features.get(node, {}).get(index, 0) for index in range(self.unique_motif_count)]
            motif_data_for_df.append(row)
            
            # print(self.features)

        # Handle empty graph case
        if not motif_data_for_df:
            print("Warning: No motif data generated (perhaps empty graph or no nodes).")
            # Ensure binned_features is still initialized for all potential nodes if needed,
            # or handle downstream logic to expect empty lists.
            # For now, proceed, binned_features will be empty lists for all nodes if num_nodes > 0.
            # If num_nodes == 0, self.binned_features is already {}
            return


        self.motifs = pd.DataFrame(motif_data_for_df)
        self.motifs.columns = ["id"] + ["role_"+str(index) for index in range(self.unique_motif_count)]

        # print("num_nodes", num_nodes)   
        # print(self.motifs)
        # exit(0)

        # Iterate through each motif type and perform quantile binning
        for index in range(self.unique_motif_count):
            col_name = "role_"+str(index)
            # Extract the feature values for this motif type across all nodes
            # .values.tolist() preserves the order from the DataFrame (which is based on node_list)
            features_for_binning = self.motifs[col_name].values.tolist()

            # Only perform qcut if there are features to bin
            if features_for_binning and sum(features_for_binning) > 0:
                # Apply log transformation
                log_features = [math.log(feature+1) for feature in features_for_binning]

                # Perform quantile binning
                # pd.qcut result's index corresponds to the input list's index (0 to num_nodes-1)
                try:
                    # Use features_for_binning as the Series to qcut to get consistent indexing
                    binned_labels = pd.qcut(pd.Series(log_features), self.args.quantiles, duplicates="drop", labels=False)
                except ValueError as e:
                     # This happens if quantiles cannot be formed (e.g., all feature values are the same)
                     print(f"Warning: Could not perform qcut for motif index {index} with quantiles {self.args.quantiles}. Error: {e}. All nodes will get the same bin (0) for this motif.")
                     binned_labels = pd.Series(np.zeros(num_nodes, dtype=int), index=range(num_nodes)) # Assign everyone to bin 0

                # Append the binned label (converted to string) to the list for each node
                # Iterate using the index 'i' from the ordered node_list
                for i, node in enumerate(node_list):
                    node_str = str(node)
                    # Use the index 'i' to access the correct binned label from binned_labels
                    # Also, handle potential missing indices from qcut if duplicates='drop'
                    binned_value = binned_labels.iloc[i] if i < len(binned_labels) else 0 # Default to bin 0 if index is missing for some reason
                    # The combined label for this motif and bin
                    combined_label = str(int(index * self.args.quantiles + binned_value))
                    self.binned_features[node_str].append(combined_label)
            else:
                 # If sum is 0 or list is empty, all nodes have 0 for this motif.
                 # They should get the same bin label for this motif. Assign bin 0.
                 print(f"Info: Motif index {index} has sum 0 or no features. Assigning bin 0 for all nodes.")
                 for node in node_list:
                      node_str = str(node)
                      combined_label = str(int(index * self.args.quantiles + 0)) # Assign to bin 0
                      self.binned_features[node_str].append(combined_label)


    def join_strings(self):
        """
        Creating string labels by joining the individual quantile labels.
        Returns {node_id_str: [combined_role_str]}
        """
        # Ensure binned_features was populated. If create_tabular_motifs was skipped
        # due to empty graph, this dict might be empty or contain empty lists.
        final_roles = {}
        node_list = list(self.graph.nodes()) # Get nodes again in consistent order
        for node in node_list:
             node_str = str(node)
             # Handle case where a node might somehow not be in binned_features (shouldn't happen with the fix)
             # or where the binned_features list is empty (e.g., if unique_motif_count is 0 or all motifs had sum 0)
             node_binned_features = self.binned_features.get(node_str, [])
             if node_binned_features:
                combined_role_string = "_".join(node_binned_features)
             else:
                combined_role_string = "no_motif_features" # Default label if no features were generated for this node

             final_roles[node_str] = [combined_role_string] # Return as list, matching original Role2Vec output format

        return final_roles

    # def factorize_string_matrix(self):
    #     """
    #     Creating string labels by factorization.
    #     """
    #     rows = [node for node, features in self.binned_features.items() for feature in features]
    #     columns = [int(feature) for node, features in self.binned_features.items() for feature in features]
    #     scores = [1 for i in range(len(columns))]
    #     row_number = max(rows)+1
    #     column_number = max(columns)+1
    #     features = csr_matrix((scores, (rows, columns)), shape=(row_number, column_number))
    #     model = NMF(n_components=self.args.factors, init="random", random_state=self.args.seed, alpha=self.args.beta)
    #     factors = model.fit_transform(features)
    #     kmeans = KMeans(n_clusters=self.args.clusters, random_state=self.args.seed).fit(factors)
    #     labels = kmeans.labels_
    #     features = {str(node): str(labels[node]) for node in self.graph.nodes()}
    #     return features
    
    def factorize_string_matrix(self):
        """
        Creating string labels by factorization.
        Handles mapping between original node IDs and contiguous indices
        for sparse matrix and clustering.
        """
        # Get the list of nodes in a consistent order for indexing
        # Convert nodes to a list and get original node IDs
        node_list = list(self.graph.nodes())
        num_nodes = len(node_list)
        # Create a mapping from original node ID to its contiguous index (0 to num_nodes-1)
        node_to_internal_index = {node: i for i, node in enumerate(node_list)}

        rows = [] # Will store internal indices (0 to num_nodes-1)
        columns = [] # Will store feature indices (based on quantile bin + motif index)
        scores = [] # Will store the value (always 1 for presence)

        # Iterate through features using the ordered node list
        # self.binned_features stores features keyed by *string* node IDs
        for node in node_list:
            node_str = str(node)
            if node_str in self.binned_features:
                features_for_node = self.binned_features[node_str]
                node_internal_index = node_to_internal_index[node]

                for feature_str in features_for_node:
                    # Ensure the feature string can be converted to an integer index
                    try:
                        feature_index = int(feature_str)
                        rows.append(node_internal_index)
                        columns.append(feature_index)
                        scores.append(1)
                    except ValueError:
                        print(f"Warning: Could not convert feature string '{feature_str}' to int for node {node}. Skipping feature.")
                        continue # Skip this feature if it's not a valid integer string


        # Handle cases where no features or nodes are present
        if num_nodes == 0 or not rows:
            print("Warning: No nodes or no motif features generated. Returning empty roles.")
            # Return empty dict or handle as appropriate for your application
            return {str(node): "no_role" for node in self.graph.nodes()}


        row_number = num_nodes # The number of nodes is the size of the first dimension
        column_number = max(columns)+1 if columns else 0 # Max feature index + 1

        # Handle case with no columns (e.g., no unique features generated)
        if column_number == 0:
            print("Warning: No unique motif feature indices found. Assigning default role.")
            return {str(node): "no_features" for node in self.graph.nodes()}


        features_matrix = csr_matrix((scores, (rows, columns)), shape=(row_number, column_number))

        # Handle case where NMF n_components > features.shape[1]
        # NMF requires n_components <= n_features (column_number)
        n_components = min(self.args.factors, column_number)
        if n_components <= 0:
            print(f"Warning: Effective number of factors ({n_components}) is zero or negative (requested {self.args.factors}, max_features {column_number}). Assigning default role.")
            return {str(node): "no_factors" for node in self.graph.nodes()}

        # Handle case where clusters > number of samples (rows = num_nodes)
        n_clusters = min(self.args.clusters, num_nodes)
        if n_clusters <= 1: # Need at least 2 clusters for meaningful clustering
            print(f"Warning: Effective number of clusters ({n_clusters}) is less than 2 (requested {self.args.clusters}, num_nodes {num_nodes}). Assigning a single default role or based on first node's cluster if possible.")
            # Assign a single role if only one cluster is possible
            return {str(node): "single_cluster" for node in self.graph.nodes()}


        try:
            # Use the adjusted number of components
            model = NMF(n_components=n_components, init="random", random_state=self.args.seed, alpha=self.args.beta)
            factors = model.fit_transform(features_matrix)

            # Use the adjusted number of clusters
            # Added n_init for robustness in KMeans initialization
            kmeans = KMeans(n_clusters=n_clusters, random_state=self.args.seed, n_init=10).fit(factors)
            labels = kmeans.labels_ # This array is indexed 0 to num_nodes-1

        except Exception as e:
            print(f"Error during NMF/KMeans for snapshot: {e}. Assigning default role.")
            # Assign a default role if clustering fails
            return {str(node): "clustering_failed" for node in self.graph.nodes()}


        # Map the cluster labels (indexed 0 to num_nodes-1) back to original node IDs
        final_features = {}
        for i, node in enumerate(node_list):
            final_features[str(node)] = str(labels[i]) # labels[i] corresponds to node_list[i]

        return final_features

    def create_string_labels(self):
        """
        Executing the whole label creation mechanism.
        """
        self.create_edge_subsets()
        self.enumerate_graphs()
        self.enumerate_categories()
        self.setup_features()
        self.create_tabular_motifs()
        if self.args.motif_compression == "string":
            features = self.join_strings()
        else:
            features = self.factorize_string_matrix()
        return features
