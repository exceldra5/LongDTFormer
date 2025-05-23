from __future__ import division, print_function
from sklearn.metrics import roc_auc_score
from sklearn.metrics import (f1_score, auc, precision_recall_curve)
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import random

np.random.seed(123)
operatorTypes = ["HAD"] # Currently only Hadamard product is implemented and used

# No changes needed here
def write_to_csv(test_results, output_name, model_name, dataset, time_steps, mod='val'):
    with open(output_name, 'a+') as f:
        for op in test_results:
            # Let's make accessing the primary AUC score more robust.
            if op in test_results and test_results[op]: # Check if list is not empty
                primary_auc_score = test_results[op][0]
                print(f"{model_name} results ({mod}) for {op}: {test_results[op]}")
                f.write(f"{dataset},{time_steps},{model_name},{op},{mod},AUC,{primary_auc_score}\n")
            else:
                print(f"Warning: No results found for operator {op} in write_to_csv.")


# No changes needed here - operates on NumPy arrays
def get_link_score(fu, fv, operator):
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == "HAD":
        return np.multiply(fu, fv)
    else:
        raise NotImplementedError

# No changes needed here - operates on lists of links and NumPy embeddings
def get_link_feats(links, source_embeddings, target_embeddings, operator):
    features = []
    for l_idx, l_val in enumerate(links):
        a, b = l_val[0], l_val[1]
        if not (0 <= a < len(source_embeddings) and 0 <= b < len(target_embeddings)):
            # print(f"Warning: Invalid node index in get_link_feats. Link: ({a},{b}), Emb shapes: {len(source_embeddings)}, {len(target_embeddings)}. Skipping.")
            pass # Or raise error, or append a default feature

        f = get_link_score(source_embeddings[a], target_embeddings[b], operator)
        features.append(f)
    return features


# No changes needed here. Operates on edge lists and NumPy embeddings.
def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds):
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])

    test_auc, test_f1_sig, test_pr_auc_sig = get_roc_score_t(test_pos, test_neg, source_embeds, target_embeds)
    val_auc, val_f1_sig, val_pr_auc_sig = get_roc_score_t(val_pos, val_neg, source_embeds, target_embeds)

    test_results['SIGMOID'].extend([test_auc, test_f1_sig, test_pr_auc_sig])
    val_results['SIGMOID'].extend([val_auc, val_f1_sig, val_pr_auc_sig])

    for operator in operatorTypes: # Only "HAD"
        train_pos_feats = np.array(get_link_feats(train_pos, source_embeds, target_embeds, operator))
        train_neg_feats = np.array(get_link_feats(train_neg, source_embeds, target_embeds, operator))
        val_pos_feats = np.array(get_link_feats(val_pos, source_embeds, target_embeds, operator))
        val_neg_feats = np.array(get_link_feats(val_neg, source_embeds, target_embeds, operator))
        test_pos_feats = np.array(get_link_feats(test_pos, source_embeds, target_embeds, operator))
        test_neg_feats = np.array(get_link_feats(test_neg, source_embeds, target_embeds, operator))
        
        # Handle empty feature sets if any split is empty
        if train_pos_feats.size == 0 or train_neg_feats.size == 0:
            print(f"Warning: Training features empty for operator {operator}. Skipping Logistic Regression.")
            # Populate with dummy results or skip operator
            val_results[operator].extend([0.0, 0.0, 0.0]) # AUC, F1, PR AUC
            test_results[operator].extend([0.0, 0.0, 0.0])
            continue


        train_pos_labels = np.ones(len(train_pos_feats)) # Use 1 for positive class
        train_neg_labels = np.zeros(len(train_neg_feats))# Use 0 for negative class (standard for scikit-learn)
        
        val_pos_labels = np.ones(len(val_pos_feats))
        val_neg_labels = np.zeros(len(val_neg_feats))
        test_pos_labels = np.ones(len(test_pos_feats))
        test_neg_labels = np.zeros(len(test_neg_feats))

        train_data = np.vstack((train_pos_feats, train_neg_feats))
        train_labels = np.concatenate((train_pos_labels, train_neg_labels))

        val_data = np.vstack((val_pos_feats, val_neg_feats))
        val_labels = np.concatenate((val_pos_labels, val_neg_labels))

        test_data = np.vstack((test_pos_feats, test_neg_feats))
        test_labels = np.concatenate((test_pos_labels, test_neg_labels))
        
        # Check if val_data or test_data is empty
        if val_data.size == 0: val_predict = np.array([])
        else:
            if train_data.size == 0:
                 print("Error: train_data is empty in evaluate_classifier. Cannot fit model.")
                 val_roc_score, val_f1_had, val_pr_had = 0.0, 0.0, 0.0
            else:
                logistic = linear_model.LogisticRegression(solver='liblinear', max_iter=1000, random_state=123) # liblinear is good for smaller datasets
                logistic.fit(train_data, train_labels)
                val_predict = logistic.predict_proba(val_data)[:, 1] # Prob of positive class (1)
                
                # ROC AUC
                if len(np.unique(val_labels)) > 1: 
                    val_roc_score = roc_auc_score(val_labels, val_predict)
                else: val_roc_score = 0.5 

                # F1 Score (binary, requires thresholding predict_proba)
                val_pred_labels_for_f1 = (val_predict >= 0.5).astype(int) # Standard threshold
                val_f1_had = f1_score(val_labels, val_pred_labels_for_f1, zero_division=0)
                
                # PR AUC
                val_ps, val_rs, _ = precision_recall_curve(val_labels, val_predict)
                val_pr_had = auc(val_rs, val_ps)

            val_results[operator].extend([val_roc_score, val_f1_had, val_pr_had])
            # val_pred_true[operator].extend(zip(val_predict, val_labels)) # Not used

        if test_data.size == 0: test_predict = np.array([])
        else:
            if train_data.size == 0: 
                 test_roc_score, test_f1_had, test_pr_had = 0.0, 0.0, 0.0
            else: # Re-use fitted logistic model if not already done
                if 'logistic' not in locals() or train_data.size == 0 : # if train_data was empty for val, logistic won't be defined
                     logistic = linear_model.LogisticRegression(solver='liblinear', max_iter=1000, random_state=123)
                     logistic.fit(train_data, train_labels) # This fit might be redundant if val_data was processed
                
                test_predict = logistic.predict_proba(test_data)[:, 1]

                if len(np.unique(test_labels)) > 1:
                    test_roc_score = roc_auc_score(test_labels, test_predict)
                else: test_roc_score = 0.5

                test_pred_labels_for_f1 = (test_predict >= 0.5).astype(int)
                test_f1_had = f1_score(test_labels, test_pred_labels_for_f1, zero_division=0)
                
                test_ps, test_rs, _ = precision_recall_curve(test_labels, test_predict)
                test_pr_had = auc(test_rs, test_ps)
            
            test_results[operator].extend([test_roc_score, test_f1_had, test_pr_had])

    return val_results, test_results #, val_pred_true, test_pred_true


# No changes needed here. Operates on edge lists and NumPy embeddings.
def get_roc_score_t(edges_pos, edges_neg, source_emb, target_emb):
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    source_emb_np = np.array(source_emb)
    target_emb_np = np.array(target_emb)
    
    if source_emb_np.size == 0 or target_emb_np.size == 0:
        # print("Warning: Empty embeddings in get_roc_score_t. Returning 0 scores.")
        return 0.0, 0.0, 0.0 # AUC, F1, PR_AUC

    adj_rec = np.dot(source_emb_np, target_emb_np.T)
    
    pred_pos = []
    if edges_pos.size > 0:
        for e in edges_pos:
            if 0 <= e[0] < adj_rec.shape[0] and 0 <= e[1] < adj_rec.shape[1]:
                pred_pos.append(sigmoid(adj_rec[e[0], e[1]]))
            # else:
                # print(f"Warning: Index out of bounds in get_roc_score_t for positive edge {e}. adj_rec shape: {adj_rec.shape}")
    
    pred_neg = []
    if edges_neg.size > 0:
        for e in edges_neg:
            if 0 <= e[0] < adj_rec.shape[0] and 0 <= e[1] < adj_rec.shape[1]:
                pred_neg.append(sigmoid(adj_rec[e[0], e[1]]))
            # else:
                # print(f"Warning: Index out of bounds in get_roc_score_t for negative edge {e}. adj_rec shape: {adj_rec.shape}")

    if not pred_pos and not pred_neg: 
        # print("Warning: No valid positive or negative predictions in get_roc_score_t.")
        return 0.0, 0.0, 0.0

    pred_all = np.hstack([pred_pos, pred_neg]) if pred_pos or pred_neg else np.array([])
    labels_all = np.hstack([np.ones(len(pred_pos)), np.zeros(len(pred_neg))]) if pred_pos or pred_neg else np.array([])

    if len(np.unique(labels_all)) < 2 or pred_all.size == 0:
        return 0.5, 0.0, 0.0 # Default AUC to 0.5 (random), F1/PR to 0

    roc_score = roc_auc_score(labels_all, pred_all)
    
    pred_labels_for_f1 = (pred_all >= 0.5).astype(int)
    f1 = f1_score(labels_all, pred_labels_for_f1, zero_division=0)

    # PR AUC
    ps, rs, _ = precision_recall_curve(labels_all, pred_all)
    pr_auc_val = auc(rs, ps)
    
    return roc_score, f1, pr_auc_val