#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coalition机制消融实验脚本。

验证DCFCL/CollEdge中coalition聚合机制的有效性。
支持 EMNIST-Letters 和 CIFAR100 两个数据集。

消融变量设置：
1. CollEdge (Full)           - 完整版本，基于相似度的动态coalition形成
2. CollEdge (FedAvg Agg)     - 用FedAvg聚合替代coalition聚合
3. CollEdge (Random Coalition) - 随机形成联盟而非基于相似度
4. CollEdge (Singleton)      - 每个客户端单独一个联盟（纯本地训练）
5. CollEdge (Global Union)   - 所有客户端作为一个大联盟

用法：
  python scripts/run_coalition_ablation.py                  # EMNIST（默认）
  python scripts/run_coalition_ablation.py --dataset cifar100
"""

import sys
import os
import copy
import logging
import random
import numpy as np
import torch
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed


def setup_logger(name, log_file=None):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def run_experiment(config, coalition_mode='full', seed=42, logger=None):
    """
    运行单个实验。
    
    Args:
        config: 配置对象
        coalition_mode: coalition模式，可选值：
            - 'full': 完整的基于相似度的coalition形成
            - 'fedavg': 用FedAvg聚合替代coalition（无coalition）
            - 'random': 随机形成coalition
            - 'singleton': 每个客户端单独一个联盟
            - 'global': 所有客户端作为一个大联盟
        seed: 随机种子
        logger: 日志记录器
    
    Returns:
        dict: 实验结果
    """
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 抑制部分日志
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)
    
    server = DCFCLServer(config, device)
    num_tasks = config.num_tasks
    rounds_per_task = config.rounds_per_task
    
    # 保存原始聚合函数
    original_aggregate_dcfcl = server._aggregate_dcfcl
    original_form_coalition_initial = server._form_coalition_initial
    original_form_coalition_dynamic = server._form_coalition_dynamic
    
    # 根据模式修改coalition聚合行为
    if coalition_mode == 'fedavg':
        # FedAvg聚合：不使用coalition，直接全局平均
        def fedavg_aggregate(train_results, global_round):
            # 聚合原型（保持原有功能）
            if config.algorithm == 'DCFCL':
                server.proto_global = server._aggregate_prototypes(train_results['proto_locals'])
                server.radius_global = server._aggregate_radius(train_results['radius_locals'])
            
            # FedAvg聚合
            server._zero_model_parameters(server.model)
            total_samples = sum(c.train_samples for c in server.clients)
            for client in server.clients:
                ratio = client.train_samples / total_samples
                server._add_parameters(client.model, ratio)
            server._broadcast_parameters()
        
        server._aggregate_dcfcl = fedavg_aggregate
        
    elif coalition_mode == 'random':
        # 随机coalition：随机将客户端分组
        def random_coalition_initial():
            n = server.num_clients
            clients = list(range(n))
            random.shuffle(clients)
            
            # 随机分组（2-4个一组）
            unions = []
            i = 0
            while i < n:
                group_size = min(random.randint(2, 4), n - i)
                unions.append(tuple(clients[i:i+group_size]))
                i += group_size
            
            server.unions = tuple(unions)
            if logger:
                logger.info(f"Random coalitions: {server.unions}")
        
        def random_coalition_dynamic():
            # 动态阶段保持随机但可能变化
            if random.random() < 0.3:  # 30%概率重新分组
                random_coalition_initial()
        
        server._form_coalition_initial = random_coalition_initial
        server._form_coalition_dynamic = random_coalition_dynamic
        
    elif coalition_mode == 'singleton':
        # 单例coalition：每个客户端单独一个联盟
        def singleton_coalition():
            server.unions = tuple((i,) for i in range(server.num_clients))
            if logger:
                logger.info(f"Singleton coalitions: {server.unions}")
        
        server._form_coalition_initial = singleton_coalition
        server._form_coalition_dynamic = singleton_coalition
        
    elif coalition_mode == 'global':
        # 全局coalition：所有客户端作为一个大联盟（等价于FedAvg但保持coalition框架）
        def global_coalition():
            server.unions = (tuple(range(server.num_clients)),)
            if logger:
                logger.info(f"Global coalition: {server.unions}")
        
        server._form_coalition_initial = global_coalition
        server._form_coalition_dynamic = global_coalition
    
    # 训练循环
    all_acc = []
    all_accs = []
    all_forget = []
    coalition_history = []
    
    for task in range(num_tasks):
        if logger:
            logger.info(f"\n{'='*60}")
            logger.info(f"Task {task}")
            logger.info(f"{'='*60}")
        
        if task > 0:
            server._update_clients_for_new_task(task)
        server._update_available_labels()
        
        for round_in_task in range(rounds_per_task):
            global_round = task * rounds_per_task + round_in_task
            
            if config.algorithm != 'Local' and global_round == 0:
                server._broadcast_parameters()
            
            train_results = server._local_training(global_round, task)
            server._server_aggregation(global_round, task, train_results)
            
            # 记录coalition结构
            if hasattr(server, 'unions') and server.unions is not None:
                coalition_history.append({
                    'round': global_round,
                    'task': task,
                    'unions': server.unions
                })
            
            accs, avg_acc, _ = server._evaluate()
            all_acc.append(avg_acc)
            
            if round_in_task == rounds_per_task - 1:
                all_accs.append(accs)
                if task > 0:
                    forget_rate = server._compute_forgetting()
                    all_forget.append(forget_rate)
                    if logger:
                        logger.info(f"Task {task} - Acc: {avg_acc:.4f}, Forget: {forget_rate:.4f}")
                else:
                    if logger:
                        logger.info(f"Task {task} - Acc: {avg_acc:.4f}")
    
    # 计算平均任务准确率
    task_end_accs = [all_acc[rounds_per_task * (t + 1) - 1] for t in range(num_tasks)]
    avg_task_acc = sum(task_end_accs) / len(task_end_accs)
    avg_forget = sum(all_forget) / len(all_forget) if all_forget else 0.0
    
    results = {
        'coalition_mode': coalition_mode,
        'final_accuracy': all_acc[-1],
        'avg_task_accuracy': avg_task_acc,
        'avg_forgetting': avg_forget,
        'task_end_accs': task_end_accs,
        'all_accuracies': all_acc,
        'all_forgetting': all_forget,
        'coalition_history': coalition_history,
    }
    
    return results


def _build_config_emnist():
    """基础配置：EMNIST-Letters（Shuffle 划分）"""
    config = Config()
    config.dataset = "EMNIST-Letters"
    config.data_split_file = "split_files/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl"
    config.num_users = 8
    config.num_tasks = 6
    config.num_rounds = 60
    config.local_epochs = 100
    config.batch_size = 64
    config.model = "cnn"
    config.lr = 1e-4
    config.weight_decay = 1e-5
    config.seed = 42
    config.rounds_per_task = config.num_rounds // config.num_tasks
    config.algorithm = "CollEdge"
    config.sw = 0.1
    config.lambda_kd = 0.2
    config.lambda_proto_aug = 2.0
    config.global_weight = 0.9
    config.ema_global = 0.9
    config.dcfcl_broadcast = 1
    config.buffer_size = 500
    config.der_alpha = 0.5
    config.der_beta = 0.5
    return config


def _build_config_cifar100():
    """基础配置：CIFAR100（random-sample 4×20 划分）"""
    config = Config()
    config.dataset = "CIFAR100"
    config.datadir = "./datasets"
    config.data_split_file = "split_files/CIFAR100_split_cn10_tn4_cet20_s2571.pkl"
    config.num_users = 10
    config.num_tasks = 4
    config.num_rounds = 40
    config.local_epochs = 50
    config.batch_size = 64
    config.model = "resnet18"
    config.feature_dim = 512
    config.lr = 0.001
    config.weight_decay = 0.001
    config.seed = 42
    config.rounds_per_task = config.num_rounds // config.num_tasks
    config.algorithm = "CollEdge"
    config.sw = 0.1
    config.lambda_kd = 0.2
    config.lambda_proto_aug = 0.1
    config.global_weight = 0.9
    config.ema_global = 0.9
    config.dcfcl_broadcast = 1
    config.buffer_size = 500
    config.der_alpha = 0.5
    config.der_beta = 0.5
    config.proto_queue_length = 100
    config._compute_derived()
    return config


def main():
    """主函数：运行coalition消融实验"""
    import argparse
    parser = argparse.ArgumentParser(description="Coalition消融实验")
    parser.add_argument("--dataset", choices=["emnist", "cifar100"], default="emnist",
                        help="数据集 (emnist|cifar100)")
    args = parser.parse_args()

    use_cifar = args.dataset == "cifar100"
    dataset_name = "CIFAR100" if use_cifar else "EMNIST-Letters"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = "_cifar100" if use_cifar else ""
    result_dir = f"results/coalition_ablation{suffix}_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)

    logger = setup_logger('CoalitionAblation', os.path.join(result_dir, 'experiment.log'))

    logger.info("=" * 80)
    logger.info(f"Coalition聚合机制消融实验  —  {dataset_name}")
    logger.info("=" * 80)
    logger.info(f"结果目录: {result_dir}")

    config = _build_config_cifar100() if use_cifar else _build_config_emnist()

    logger.info(f"num_classes: {config.num_classes}, rounds_per_task: {config.rounds_per_task}")

    ablation_modes = [
        ('full',      'CollEdge (完整版本，基于相似度的动态coalition)'),
        ('fedavg',    'CollEdge (FedAvg聚合，无coalition)'),
        ('random',    'CollEdge (随机coalition)'),
        ('singleton', 'CollEdge (Singleton，每个客户端单独一个联盟)'),
        ('global',    'CollEdge (Global，所有客户端一个大联盟)'),
    ]

    results = {}

    for mode, description in ablation_modes:
        logger.info(f"\n{'='*60}\n实验: {description}\n模式: {mode}\n{'='*60}")

        exp_config = copy.deepcopy(config)
        exp_results = run_experiment(exp_config, coalition_mode=mode, seed=42, logger=logger)
        results[mode] = {'description': description, **exp_results}

        logger.info(f"\n结果:")
        logger.info(f"  最终准确率:     {exp_results['final_accuracy']*100:.2f}%")
        logger.info(f"  平均任务准确率: {exp_results['avg_task_accuracy']*100:.2f}%")
        logger.info(f"  平均遗忘率:     {exp_results['avg_forgetting']*100:.2f}%")
        logger.info(f"  各任务准确率:   {[f'{a*100:.1f}%' for a in exp_results['task_end_accs']]}")

    # ── 汇总 ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 80)
    logger.info("消融实验结果对比")
    logger.info("=" * 80)
    logger.info(f"\n{'模式':<25} {'最终准确率':>12} {'平均任务准确率':>15} {'平均遗忘率':>12}")
    logger.info("-" * 70)
    for mode, data in results.items():
        logger.info(f"{mode:<25} {data['final_accuracy']*100:>11.2f}%"
                    f" {data['avg_task_accuracy']*100:>14.2f}%"
                    f" {data['avg_forgetting']*100:>11.2f}%")

    baseline = results['full']
    logger.info("\n" + "-" * 70)
    logger.info("相对于完整版本的变化")
    logger.info("-" * 70)
    for mode, data in results.items():
        if mode == 'full':
            continue
        acc_diff    = (data['final_accuracy']    - baseline['final_accuracy'])    * 100
        avg_diff    = (data['avg_task_accuracy'] - baseline['avg_task_accuracy']) * 100
        forget_diff = (data['avg_forgetting']    - baseline['avg_forgetting'])    * 100
        logger.info(f"{mode:<25} {acc_diff:>+11.2f}% {avg_diff:>+14.2f}% {forget_diff:>+11.2f}%")

    # ── 保存 ─────────────────────────────────────────────────────────────────
    import json
    for mode in results:
        results[mode]['coalition_history'] = [
            {**h, 'unions': [list(u) for u in h['unions']]}
            for h in results[mode]['coalition_history']
        ]

    json_path = os.path.join(result_dir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n实验完成！结果已保存到: {result_dir}")
    return results


if __name__ == '__main__':
    main()
