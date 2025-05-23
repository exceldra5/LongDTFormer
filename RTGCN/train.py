import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 
import argparse
# import networkx as nx 
import pandas as pd
# import scipy # DO NOT USE THIS!!! This is non deterministic 
from torch.utils.data import DataLoader
import torch
import torch.nn as nn 
import numpy as np

from utils.preprocess import load_graphs, get_context_pairs, get_evaluation_data, get_evaluation_classification_data
from utils.minibatch import MyDataset
from utils.utilities import to_device 
from eval.link_prediction import evaluate_classifier
from models.model import RTGCN 
from utils1 import ( 
    gen_attribute_hg, 
    cross_role_hypergraphn_nodes, 
    hypergraph_role_set, 
    # scipy_sparse_mat_to_torch_sparse_tensor, 
    evaluate_node_classification, 
    set_random_seed # For reproducibility
)

set_random_seed(0)

torch.autograd.set_detect_anomaly(True) # Keep for debugging

def numpy_adj_to_torch_sparse_coo(adj_matrix_np, device='cpu'):
    """Converts a NumPy adjacency matrix to a PyTorch sparse COO tensor."""
    if not isinstance(adj_matrix_np, np.ndarray):
        adj_matrix_np = np.array(adj_matrix_np)
    
    rows, cols = np.where(adj_matrix_np > 0) # Assuming unweighted, get indices
    values = adj_matrix_np[rows, cols] # Get the actual weights/values

    indices = torch.from_numpy(np.vstack((rows, cols))).long()
    values_tensor = torch.from_numpy(values).float() # Assuming float weights
    shape = torch.Size(adj_matrix_np.shape)
    
    return torch.sparse_coo_tensor(indices, values_tensor, shape).to(device)


def inductive_graph(graph_former_repr, graph_later_repr):
    """
    Creates an inductive graph representation.
    Nodes from graph_later, edges from graph_former (filtered by later nodes).
    graph_former_repr: {'edge_index': tensor, 'num_nodes': int, ...}
    graph_later_repr: {'edge_index': tensor, 'num_nodes': int, ...}
    Returns: new_graph_repr {'edge_index': tensor, 'num_nodes': int}
    """
    nodes_later_count = graph_later_repr['num_nodes']
    
    # Edges from former graph
    edge_index_former = graph_former_repr['edge_index']
    # Optional: edge_attr_former if present and needed
    
    if edge_index_former.numel() == 0: # No edges in former graph
        return {
            'edge_index': torch.empty((2,0), dtype=torch.long, device=edge_index_former.device),
            'num_nodes': nodes_later_count
            # Add 'edge_attr' if it's part of the representation
        }

    # Filter edges: both nodes must be within the count of nodes_later
    mask_u = edge_index_former[0, :] < nodes_later_count
    mask_v = edge_index_former[1, :] < nodes_later_count
    combined_mask = mask_u & mask_v
    
    inductive_edge_index = edge_index_former[:, combined_mask]
    # inductive_edge_attr = edge_attr_former[combined_mask] if edge_attr_former is not None else None

    return {
        'edge_index': inductive_edge_index,
        'num_nodes': nodes_later_count
        # 'edge_attr': inductive_edge_attr
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_steps', type=int, default=3)
    parser.add_argument('--dataset', type=str, default='DBLP3')
    parser.add_argument('--GPU_ID', type=int, default=0) 
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    # parser.add_argument('--featureless', type=bool, default=False)
    parser.add_argument("--early_stop", type=int, default=50)
    parser.add_argument('--node_num', type=int, default=4257) 
    parser.add_argument('--input_dim', type=int, default=100) 
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--output_dim', type=int, default=24)
    parser.add_argument('--task', type=str, default='link prediction', choices=['link prediction', 'node classification'])
    parser.add_argument('--residual', type=bool, default=True) # Was nargs='?'
    parser.add_argument('--neg_sample_size', type=int, default=10)
    parser.add_argument('--walk_len', type=int, default=20)
    parser.add_argument('--neg_weight', type=float, default=10)
    parser.add_argument('--loss_weight', type=float, default=1.0) # Role loss weight
    parser.add_argument('--learning_rate', type=float, default=0.008)
    parser.add_argument('--weight_decay', type=float, default=0.0003)
    parser.add_argument('--window', type=int, default=-1)
    
    args = parser.parse_args()
    print(args)

    # Setup device
    if torch.cuda.is_available() and args.GPU_ID >= 0:
        device = torch.device(f"cuda:{args.GPU_ID}")
    else:
        device = torch.device("cpu")
    args.device = device 
    print(f"Using device: {device}")

    # Load structural role data (Pandas DataFrame from .pkl)
    role_file_prefix = args.dataset # e.g. DBLP3
    role_path = f'./data/{args.dataset}/{role_file_prefix}_wl_nc.pkl'

    train_role_graph_df = pd.read_pickle(role_path) # Dict of dicts {ts: {role_id: [nodes]}}

    list_loss_role = [] 
    for t in range(args.time_steps): 
        if t in train_role_graph_df:
            list_g_for_ts = []
            for role_id in train_role_graph_df[t]:
                nodes_in_role = list(map(int, train_role_graph_df[t][role_id]))
                if nodes_in_role: # Only add if role is not empty
                    list_g_for_ts.append(torch.tensor(nodes_in_role, dtype=torch.long).to(device))
            list_loss_role.append(list_g_for_ts)
        else:
            list_loss_role.append([]) 
            
    graphs_repr, raw_adjs_np_list, raw_features_list, data_npz_content = \
        load_graphs(args.dataset, args.time_steps, device=device)

    # Prepare Data_dblp for the model input
    # Data_dblp[0]: list of feature tensors per timestep
    # Data_dblp[1]: list of edge_index tensors per timestep
    # Data_dblp[2]: list of sparse adjacency matrix tensors per timestep (for MatGRUCell)
    
    Data_dblp = []
    model_features_input = [torch.from_numpy(feat_np).float().to(device) for feat_np in raw_features_list]
    Data_dblp.append(model_features_input)

    model_edge_indices_input = [g_repr['edge_index'] for g_repr in graphs_repr]
    Data_dblp.append(model_edge_indices_input)
    
    model_adj_sparse_input = [numpy_adj_to_torch_sparse_coo(adj_np, device=device) for adj_np in raw_adjs_np_list]
    Data_dblp.append(model_adj_sparse_input)

    labels_np = data_npz_content['labels'] # For node classification

    # Construct role hypergraphs 
    train_hypergraph_laplacians = [] 
    cross_role_hyper_incidences = [] 
    cross_role_laplacians = []      

    # Role_set and Cross_role_Set are lists of unique role IDs
    Role_set, Cross_role_Set = hypergraph_role_set(train_role_graph_df, args.time_steps)
    H_prev_ts_roles = None # To store H from gen_attribute_hg of t-1

    set_random_seed(0)
    for i in range(args.time_steps):
        if i not in train_role_graph_df: # Handle missing timesteps in role data
            # Append placeholder sparse tensors if a timestep is missing
            # This requires knowing num_nodes and len(Role_set)
            num_nodes_current = args.node_num
            num_hyperedges_role = len(Role_set)
            empty_laplacian = torch.sparse_coo_tensor(
                torch.empty((2,0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float, device=device),
                (num_nodes_current, num_nodes_current), device=device
            )
            train_hypergraph_laplacians.append(empty_laplacian)
            
            if i > 0: # For cross-role parts
                num_hyperedges_cross = len(Cross_role_Set)
                empty_incidence_cross = torch.sparse_coo_tensor(
                     torch.empty((2,0), dtype=torch.long, device=device),
                     torch.empty((0,), dtype=torch.float, device=device),
                    (num_nodes_current, num_hyperedges_cross), device=device
                )
                cross_role_hyper_incidences.append(empty_incidence_cross)
                cross_role_laplacians.append(empty_laplacian) # Lap of prev cross-role
            
            H_prev_ts_roles = None # Reset as this timestep was missing
            continue


        Role_hyper_obj, H_role_incidence = gen_attribute_hg(
            args.node_num, train_role_graph_df[i], Role_set, X=None, device=device
        )
        train_hypergraph_laplacians.append(Role_hyper_obj.laplacian())

        if i > 0 and (i-1) in train_role_graph_df:
            _, H_role_incidence_for_cross_combination = gen_attribute_hg(
                 args.node_num, train_role_graph_df[i], Cross_role_Set, X=None, device=device # Use Cross_role_Set for H part of sum
            )

            prev_role_hyper_obj, combined_cross_role_incidence = cross_role_hypergraphn_nodes(
                args.node_num, 
                H_current_roles=H_role_incidence_for_cross_combination, # H_t based on Cross_role_Set
                role_dict1=train_role_graph_df[i-1], # Roles at t-1
                role_dict2=train_role_graph_df[i],   # Roles at t (used for filtering common roles with t-1)
                Cross_role_Set=Cross_role_Set,      # Defines hyperedges for H1 (t-1 -> t)
                w_decay_factor=-11, delta_t=1, X=None, device=device
            )
            cross_role_hyper_incidences.append(combined_cross_role_incidence) # H_t + H1
            cross_role_laplacians.append(prev_role_hyper_obj.laplacian()) # Laplacian of H1
        elif i > 0: # i-1 was not in train_role_graph_df, append empty
            num_nodes_current = args.node_num
            num_hyperedges_cross = len(Cross_role_Set)
            empty_incidence_cross = torch.sparse_coo_tensor(
                 torch.empty((2,0), dtype=torch.long, device=device),
                 torch.empty((0,), dtype=torch.float, device=device),
                (num_nodes_current, num_hyperedges_cross), device=device
            )
            empty_laplacian_cross = torch.sparse_coo_tensor(
                torch.empty((2,0), dtype=torch.long, device=device),
                torch.empty((0,), dtype=torch.float, device=device),
                (num_nodes_current, num_nodes_current), device=device
            )
            cross_role_hyper_incidences.append(empty_incidence_cross)
            cross_role_laplacians.append(empty_laplacian_cross)


    # Context pairs and evaluation data splits
    context_pairs_train = get_context_pairs(graphs_repr, raw_adjs_np_list, args.time_steps,
                                            num_walks=10, walk_len=args.walk_len) # p,q default to 1
    
    train_edges_pos, train_edges_neg, val_edges_pos, val_edges_neg, \
    test_edges_pos, test_edges_neg = get_evaluation_data(graphs_repr, args.time_steps)
    
    print(f"No. Train: Pos={len(train_edges_pos)}, Neg={len(train_edges_neg)} \n"
          f"No. Val: Pos={len(val_edges_pos)}, Neg={len(val_edges_neg)} \n"
          f"No. Test: Pos={len(test_edges_pos)}, Neg={len(test_edges_neg)}")

    train_nodes_cls_splits = None
    if args.task == 'node classification':
        train_nodes_cls_splits = get_evaluation_classification_data(args.dataset, args.node_num, args.time_steps)

    # Inductive graph modification (last snapshot)
    if args.time_steps >= 2:
        idx_former = args.time_steps - 2
        idx_later = args.time_steps - 1
        
        inductive_snapshot_repr = inductive_graph(graphs_repr[idx_former], graphs_repr[idx_later])
        graphs_repr[idx_later] = inductive_snapshot_repr 
        
        # Update corresponding entries in Data_dblp if model uses them directly from there post-inductive step
        Data_dblp[1][idx_later] = inductive_snapshot_repr['edge_index']
        
        # Convert inductive_edge_index to a sparse adj matrix
        num_nodes_inductive = inductive_snapshot_repr['num_nodes']
        inductive_adj_values = torch.ones(inductive_snapshot_repr['edge_index'].shape[1], device=device)
        inductive_adj_sparse = torch.sparse_coo_tensor(
            inductive_snapshot_repr['edge_index'], 
            inductive_adj_values, 
            (num_nodes_inductive, num_nodes_inductive)
        ).coalesce() 
        Data_dblp[2][idx_later] = inductive_adj_sparse
    else:
        print("Warning: Not enough time_steps for inductive graph setup.")


    # Build DataLoader and Model
    dataset = MyDataset(args, graphs_repr, raw_features_list, raw_adjs_np_list, context_pairs_train)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            num_workers=0, # Set to >0 for parallel data loading if beneficial
                            collate_fn=MyDataset.collate_fn,
                            pin_memory=False)
    
    model = RTGCN(act=nn.ELU(),
                  n_node=args.node_num,
                  input_dim=args.input_dim,
                  output_dim=args.output_dim,
                  hidden_dim=args.hidden_dim,
                  time_step=args.time_steps,
                  neg_weight=args.neg_weight,
                  loss_weight=args.loss_weight,
                  attn_drop=0.0,
                  residual=args.residual,
                  role_num=len(Role_set), # Number of unique roles
                  cross_role_num=len(Cross_role_Set) # Num cross-roles (can be same as Role_set)
                 ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    # Training loop
    best_epoch_val_auc = 0
    best_val_results_dict = {}
    best_test_results_dict = {}
    
    final_emb_for_best_val = None
    final_all_embs_for_best_val = None
    patient_count = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss_list = []
        for feed_dict_batch in dataloader:
            opt.zero_grad()
            # RTGCN.get_loss expects data (Data_dblp), hypergraph_lap, cross_role_inc, cross_role_lap
            loss = model.get_loss(feed_dict_batch, 
                                  Data_dblp, 
                                  train_hypergraph_laplacians, 
                                  cross_role_hyper_incidences, 
                                  cross_role_laplacians, 
                                  list_loss_role)
            print('Loss:', loss.item()) # Debugging print, can be removed later
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Epoch {epoch}, NaN/Inf loss detected. Stopping training.")
                epoch_loss_list.append(float('inf'))
                break 

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Optional gradient clipping
            opt.step()
            epoch_loss_list.append(loss.item())
        
        if not epoch_loss_list or epoch_loss_list[-1] == float('inf'): # Broke due to bad loss
            print("Exiting due to NaN/Inf loss in training.")
            break

        avg_epoch_loss = np.mean(epoch_loss_list)
        # dataset.test_reset() 

        # Validation
        model.eval()
        with torch.no_grad():
            all_embs_val = model.forward(Data_dblp, 
                                         train_hypergraph_laplacians, 
                                         cross_role_hyper_incidences, 
                                         cross_role_laplacians)
            
            if args.time_steps >=2 :
                emb_for_eval = all_embs_val[args.time_steps - 2].cpu().numpy()
            else: # Not enough timesteps for standard eval, use last available
                emb_for_eval = all_embs_val[-1].cpu().numpy()


            val_results, test_results_at_val_time = evaluate_classifier(
                train_edges_pos, train_edges_neg,
                val_edges_pos, val_edges_neg,
                test_edges_pos, test_edges_neg,
                emb_for_eval, emb_for_eval # source and target embeddings are the same
            )
        
        current_val_auc = val_results["HAD"][0] # Assuming HAD operator and AUC is first metric
        current_test_auc_at_val_time = test_results_at_val_time["HAD"][0]

        print(f"Epoch {epoch:<3}, Loss = {avg_epoch_loss:.3f}, Val AUC {current_val_auc:.4f}, Test AUC (at val time) {current_test_auc_at_val_time:.4f}")

        if current_val_auc > best_epoch_val_auc:
            best_epoch_val_auc = current_val_auc
            best_val_results_dict = val_results
            best_test_results_dict = test_results_at_val_time # Store test results from this best val epoch
            
            final_emb_for_best_val = emb_for_eval
            final_all_embs_for_best_val = [e.clone().detach() for e in all_embs_val] # Store all timesteps

            # torch.save(model.state_dict(), "./model_checkpoints/model.pt") # Save model
            patient_count = 0
        else:
            patient_count += 1
            if patient_count > args.early_stop:
                print(f"Early stopping at epoch {epoch} due to no improvement in Val AUC for {args.early_stop} epochs.")
                break
    
    # Final Test using the best model (based on validation)
    print("\n--- Final Test Results (using model from best validation epoch) ---")
    if final_emb_for_best_val is not None:
        # Re-evaluate on test set using the embeddings from the best validation epoch
        
        # Using best_val_results_dict and best_test_results_dict directly is fine here.
        final_val_auc = best_val_results_dict.get("HAD", [0])[0]
        final_test_auc = best_test_results_dict.get("HAD", [0])[0]
        final_test_f1 = best_test_results_dict.get("HAD", [0,0,0])[1] # F1
        final_test_pr_auc = best_test_results_dict.get("HAD", [0,0,0])[2] # PR-AUC

        print(f"Best Val AUC: {final_val_auc:.4f}")
        print(f"Corresponding Test AUC: {final_test_auc:.4f}, F1: {final_test_f1:.4f}, PR-AUC: {final_test_pr_auc:.4f}")

        if args.task == 'node classification' and train_nodes_cls_splits and final_all_embs_for_best_val:
            # Use embeddings from the *last* timestep of the best validation model state
            emb_for_node_cls = final_all_embs_for_best_val[-1].cpu().numpy()
            labels_for_cls = torch.from_numpy(labels_np).float().to(device) 

            avg_accs, temp_accs, avg_aucs, temp_aucs = evaluate_node_classification(
                emb_for_node_cls, 
                labels_for_cls.cpu().numpy(), # eval func expects numpy labels
                train_nodes_cls_splits,
                device=device
            )
            if avg_accs and len(avg_accs) >=3 and avg_aucs and len(avg_aucs) >=3: # Check if results are available for typical ratios
                print(f"Node Classification ACC (ratios 0.3, 0.5, 0.7): "
                      f"[{avg_accs[0]:.4f}, {avg_accs[1]:.4f}, {avg_accs[2]:.4f}]")
                # print(f"All ACCs per ts/ratio: {temp_accs}")
                print(f"Node Classification AUC (ratios 0.3, 0.5, 0.7): "
                      f"[{avg_aucs[0]:.4f}, {avg_aucs[1]:.4f}, {avg_aucs[2]:.4f}]")
                # print(f"All AUCs per ts/ratio: {temp_aucs}")
            else:
                print("Node classification results not fully available for typical ratios.")
                print(f"Avg ACCs: {avg_accs}, Avg AUCs: {avg_aucs}")

    else:
        print("No best model found (e.g., training did not run or complete).")