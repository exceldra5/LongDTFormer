import logging
import time
import sys
import os
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn

from models.DTFormer import DTFormer
from models.modules import MergeLayer
from utils.utils import set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction, test_model_link_prediction
from utils.metrics import get_link_prediction_metrics
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data, load_dblp3_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    feat_using = []
    if args.using_time_feat:
        feat_using.append('time_feat')
    if args.using_intersect_feat:
        feat_using.append('intersect_feat')
    if args.using_snapshot_feat:
        feat_using.append('snapshot_feat')
    if args.using_snap_counts:
        feat_using.append('snap_counts')

    data_snapshots_num = {'bitcoinalpha': 274,
                          'bitcoinotc': 279,
                          'CollegeMsg': 29,
                          'reddit-body': 178,
                          'reddit-title': 178,
                          'mathoverflow': 2350,
                          'email-Eu-core': 803,
                          'DBLP3': 1000}

    # get data for training, validation and testing
    if args.dataset_name == 'DBLP3':
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _, node_snap_counts = \
            load_dblp3_data()
    else:
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, _, _, node_snap_counts = \
            get_link_prediction_data(dataset_name=args.dataset_name, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
                                     num_snapshots=data_snapshots_num[args.dataset_name])

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                                 dst_node_ids=train_data.dst_node_ids)
    val_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids,
                                               seed=0)
    test_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=full_data.src_node_ids,
                                                dst_node_ids=full_data.dst_node_ids, seed=2)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=False)
    val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                              batch_size=args.batch_size, shuffle=False)
    test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                               batch_size=args.batch_size, shuffle=False)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):

        set_random_seed(seed=run)

        args.seed = run
        args.save_model_name = f'{args.model_name}_seed{args.seed}'

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/", exist_ok=True)
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log")
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        dynamic_backbone = DTFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    node_snap_counts=node_snap_counts, neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, channel_embedding_dim=args.channel_embedding_dim,
                                    patch_size=args.patch_size,
                                    num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                    max_input_sequence_length=args.max_input_sequence_length, device=args.device,
                                    feat_using=feat_using,
                                    num_patch_size=args.num_patch_size, intersect_mode=args.intersect_mode)

        link_predictor = MergeLayer(input_dim1=node_raw_features.shape[1], input_dim2=node_raw_features.shape[1],
                                    hidden_dim=node_raw_features.shape[1], output_dim=1)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        # 체크포인트 저장 경로 추가
        checkpoint_folder = f"./checkpoints/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        os.makedirs(checkpoint_folder, exist_ok=True)

        early_stopping = EarlyStopping(patience=args.patience, save_model_folder=save_model_folder,
                                       save_model_name=args.save_model_name, logger=logger, model_name=args.model_name)

        loss_func = nn.BCELoss()

        # 체크포인트 로드 시도
        start_epoch = 0
        best_val_metric = float('inf')
        if os.path.exists(os.path.join(checkpoint_folder, 'latest_checkpoint.pt')):
            try:
                # weights_only=False로 시도
                checkpoint = torch.load(os.path.join(checkpoint_folder, 'latest_checkpoint.pt'), weights_only=False)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_metric = checkpoint.get('best_val_metric', float('inf'))
                logger.info(f'체크포인트에서 복구: epoch {start_epoch}, best_val_metric: {best_val_metric}')
                
                # 학습이 완료된 경우 테스트만 실행
                if start_epoch >= args.num_epochs:
                    logger.info('학습이 완료되어 테스트만 실행합니다.')
                    test_losses, test_metrics = test_model_link_prediction(model=model,
                                                                         neighbor_sampler=full_neighbor_sampler,
                                                                         evaluate_idx_data_loader=test_idx_data_loader,
                                                                         evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                                         evaluate_data=test_data,
                                                                         loss_func=loss_func)
                    
                    logger.info(f'test loss: {np.mean(test_losses):.4f}')
                    for metric_name in test_metrics[0].keys():
                        average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
                        logger.info(f'test {metric_name}, {average_test_metric:.4f}')
                    
                    # 결과 저장
                    result_json = {
                        "test metrics": {metric_name: f'{np.mean([test_metric[metric_name] for test_metric in test_metrics]):.4f}' 
                                      for metric_name in test_metrics[0].keys()},
                    }
                    result_json = json.dumps(result_json, indent=4)
                    
                    save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
                    os.makedirs(save_result_folder, exist_ok=True)
                    save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")
                    
                    with open(save_result_path, 'w') as file:
                        file.write(result_json)
                    
                    sys.exit()
                
            except Exception as e:
                logger.warning(f'체크포인트 로드 실패: {str(e)}')
                logger.info('새로운 학습을 시작합니다.')
                start_epoch = 0
                best_val_metric = float('inf')

        for epoch in range(start_epoch, args.num_epochs):
            model.train()
            # training, only use training graph
            model[0].set_neighbor_sampler(train_neighbor_sampler)

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120, dynamic_ncols=True)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids, batch_snapshots = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices], \
                    train_data.snapshots[train_data_indices]

                _, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                batch_neg_src_node_ids = batch_src_node_ids

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      snapshots=batch_snapshots)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      snapshots=batch_snapshots)

                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = model[1](input_1=batch_src_node_embeddings,
                                                  input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
                negative_probabilities = model[1](input_1=batch_neg_src_node_embeddings,
                                                  input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()

                predicts = torch.cat([positive_probabilities, negative_probabilities], dim=0)
                labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                                   dim=0)

                loss = loss_func(input=predicts, target=labels)

                train_losses.append(loss.item())

                train_metrics.append(get_link_prediction_metrics(predicts=predicts, labels=labels))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

            # 매 에포크마다 체크포인트 저장
            try:
                checkpoint = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_metric': best_val_metric
                }
                torch.save(checkpoint, os.path.join(checkpoint_folder, 'latest_checkpoint.pt'), weights_only=False)
            except Exception as e:
                logger.warning(f'체크포인트 저장 실패: {str(e)}')

            val_losses, val_metrics = evaluate_model_link_prediction(model=model,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=val_neg_edge_sampler,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func)

            # 현재 validation metric 계산
            current_val_metric = np.mean(val_losses)
            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                # best model 저장
                torch.save(checkpoint, os.path.join(checkpoint_folder, 'best_model.pt'))

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(val_metric_indicator, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f'get final performance on dataset {args.dataset_name}...')

        test_losses, test_metrics = test_model_link_prediction(model=model,
                                                               neighbor_sampler=full_neighbor_sampler,
                                                               evaluate_idx_data_loader=test_idx_data_loader,
                                                               evaluate_neg_edge_sampler=test_neg_edge_sampler,
                                                               evaluate_data=test_data,
                                                               loss_func=loss_func)

        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(f'Run {run + 1} cost {single_run_time:.2f} seconds.')

        test_metric_all_runs.append(test_metric_dict)

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)

        # save model result
        result_json = {
            "test metrics": {metric_name: f'{test_metric_dict[metric_name]:.4f}' for metric_name in test_metric_dict},
        }
        result_json = json.dumps(result_json, indent=4)

        save_result_folder = f"./saved_results/{args.model_name}/{args.dataset_name}"
        os.makedirs(save_result_folder, exist_ok=True)
        save_result_path = os.path.join(save_result_folder, f"{args.save_model_name}.json")

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    logger.info(f'metrics over {args.num_runs} runs:')

    for metric_name in test_metric_all_runs[0].keys():
        logger.info(
            f'test {metric_name}, {[test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]}')
        logger.info(
            f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
            f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')

    sys.exit()
