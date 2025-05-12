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


def _process_graphlet_batch(graphlet_node_lists_batch, main_graph, size, interesting_graphs, categories, unique_motif_count):
    batch_aggregated_counts = {}
    graphs_for_size = interesting_graphs[size]

    for nodes_list_item in tqdm(graphlet_node_lists_batch, leave=False, desc=f"Worker (Size {size})"):
        sub_gr = main_graph.subgraph(nodes_list_item)

        for index, graph in enumerate(graphs_for_size):
            if nx.is_isomorphic(sub_gr, graph):
                for node in sub_gr.nodes():
                    node_degree_in_subgraph = sub_gr.degree(node)
                    if node_degree_in_subgraph in categories[size][index]:
                        orbit_id = categories[size][index][node_degree_in_subgraph]
                        if node not in batch_aggregated_counts:
                            batch_aggregated_counts[node] = {i: 0 for i in range(unique_motif_count)}
                        batch_aggregated_counts[node][orbit_id] += 1
                break

    return batch_aggregated_counts


class MotifCounterMachine(object):
    def __init__(self, graph, args):
        self.graph = graph
        self.args = args
        if args.num_jobs > 1:
            self.n_jobs = args.num_jobs
        else:
            self.n_jobs = os.cpu_count() // 2 - 4 if os.cpu_count() > 1 else 1


    def create_edge_subsets(self):
        self.edge_subsets = dict()
        subsets = [[edge[0], edge[1]] for edge in self.graph.edges()]
        self.edge_subsets[2] = subsets
        unique_subsets = dict()
        for i in range(3, self.args.graphlet_size+1):
            for subset in tqdm(subsets, desc=f"Generating Subsets Size {i-1}"):
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
        graphs = graph_atlas_g()
        self.interesting_graphs = {i: [] for i in range(2, self.args.graphlet_size+1)}
        for graph in graphs:
            if graph.number_of_nodes() > 1 and  graph.number_of_nodes() < self.args.graphlet_size+1:
                if nx.is_connected(graph):
                    self.interesting_graphs[graph.number_of_nodes()].append(graph)

    def enumerate_categories(self):
        main_index = 0
        self.categories = dict()
        for size, graphs in self.interesting_graphs.items():
            self.categories[size] = dict()
            for index, graph in enumerate(graphs):
                self.categories[size][index] = dict()
                degrees = sorted(list(set([graph.degree(node) for node in graph.nodes()])))
                for degree in degrees:
                    self.categories[size][index][degree] = main_index
                    main_index = main_index + 1
        self.unique_motif_count = main_index

    def setup_features(self):
        self.features = {n: {i: 0 for i in range(self.unique_motif_count)} for n in self.graph.nodes()}

        print("\nCalculating graphlet orbit orbit counts...")

        for size, node_lists in self.edge_subsets.items():
            print(f"Processing graphlets of size {size} ({len(node_lists)} unique instances)...")

            if not node_lists:
                print(f"No unique graphlets of size {size} found. Skipping.")
                continue

            chunk_size = math.ceil(len(node_lists) / self.n_jobs)
            chunk_size = max(1, chunk_size)

            batches = [node_lists[i:i + chunk_size] for i in range(0, len(node_lists), chunk_size)]

            # Set verbose=0 in Parallel to avoid joblib's default verbosity, letting inner tqdm handle progress visualization
            results = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0, timeout=None)(
                delayed(_process_graphlet_batch)(
                    batch,
                    self.graph,
                    size,
                    self.interesting_graphs,
                    self.categories,
                    self.unique_motif_count
                ) for batch in tqdm(batches, desc=f"Size {size} Batches")
            )

            print(f"Aggregating results for size {size}...")
            for batch_aggregated_counts in tqdm(results, desc=f"Aggregating Size {size}"):
                for node_id, orbit_counts in batch_aggregated_counts.items():
                    for orbit_id, count in orbit_counts.items():
                         if node_id in self.features and 0 <= orbit_id < self.unique_motif_count:
                            self.features[node_id][orbit_id] += count

        print("Graphlet orbit counting complete.")


    def create_tabular_motifs(self):
        node_list = list(self.graph.nodes())
        num_nodes = len(node_list)

        self.binned_features = {str(node): [] for node in node_list}

        motif_data_for_df = []
        for node in node_list:
            node_str = str(node)
            row = [node] + [self.features.get(node, {}).get(index, 0) for index in range(self.unique_motif_count)]
            motif_data_for_df.append(row)

        if not motif_data_for_df:
            print("Warning: No motif data generated.")
            return

        self.motifs = pd.DataFrame(motif_data_for_df)
        self.motifs.columns = ["id"] + ["role_"+str(index) for index in range(self.unique_motif_count)]

        for index in range(self.unique_motif_count):
            col_name = "role_"+str(index)
            features_for_binning = self.motifs[col_name].values.tolist()

            if features_for_binning and sum(features_for_binning) > 0:
                log_features = [math.log(feature+1) for feature in features_for_binning]

                try:
                    binned_labels = pd.qcut(pd.Series(log_features), self.args.quantiles, duplicates="drop", labels=False)
                except ValueError as e:
                     print(f"Warning: Could not perform qcut for motif index {index}. Error: {e}. All nodes will get the same bin (0) for this motif.")
                     binned_labels = pd.Series(np.zeros(num_nodes, dtype=int), index=range(num_nodes))

                for i, node in enumerate(node_list):
                    node_str = str(node)
                    binned_value = binned_labels.iloc[i] if i < len(binned_labels) else 0
                    combined_label = str(int(index * self.args.quantiles + binned_value))
                    self.binned_features[node_str].append(combined_label)
            else:
                 print(f"Info: Motif index {index} has sum 0 or no features. Assigning bin 0 for all nodes.")
                 for node in node_list:
                      node_str = str(node)
                      combined_label = str(int(index * self.args.quantiles + 0))
                      self.binned_features[node_str].append(combined_label)


    def join_strings(self):
        """
        Creating string labels by joining the individual quantile labels.
        Returns {node_id_str: [combined_role_str]}
        """
        final_roles = {}
        node_list = list(self.graph.nodes())
        for node in node_list:
             node_str = str(node)
             node_binned_features = self.binned_features.get(node_str, [])
             if node_binned_features:
                combined_role_string = "_".join(node_binned_features)
             else:
                combined_role_string = "no_motif_features"

             final_roles[node_str] = [combined_role_string]

        return final_roles

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