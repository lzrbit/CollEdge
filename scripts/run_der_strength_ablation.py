#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DER++强度 vs Coalition机制消融实验

假设：DER++太强（0%遗忘）掩盖了Coalition的价值。
验证：通过降低buffer_size，让遗忘出现，观察Coalition机制是否能提供更好的保护。

实验设计：
- 数据集：EMNIST-Letters (Shuffle模式，异构数据)
- buffer_size: 0 (无DER++), 50, 100, 200, 500
- 每种buffer_size测试两种聚合策略：full (动态coalition) vs fedavg (全局聚合)
- 预期：buffer_size越小，遗忘越多，full vs fedavg的差异越大
"""

import sys
import os
import copy
import logging
import random
import numpy as np
import torch
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed


def setup_logger(name, log_file=None):
    """设置日志记录器"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # 清除已有handlers
    
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
        coalition_mode: coalition模式
            - 'full': 完整的基于相似度的coalition形成
            - 'fedavg': 用FedAvg聚合替代coalition（无coalition）
        seed: 随机种子
        logger: 日志记录器
    """
    setup_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 抑制部分日志
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)
    
    server = DCFCLServer(config, device)
    num_tasks = config.num_tasks
    rounds_per_task = config.rounds_per_task
    
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
    
    # 训练循环
    all_acc = []
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
            if coalition_mode == 'full' and hasattr(server, 'unions') and server.unions is not None:
                coalition_history.append({
                    'round': global_round,
                    'task': task,
                    'unions': [list(u) for u in server.unions]
                })
            
            accs, avg_acc, _ = server._evaluate()
            all_acc.append(avg_acc)
            
            if round_in_task == rounds_per_task - 1:
                if task > 0:
                    forget_rate = server._compute_forgetting()
                    all_forget.append(forget_rate)
                    if logger:
                        logger.info(f"Task {task} - Acc: {avg_acc:.4f}, Forget: {forget_rate:.4f}")
                else:
                    if logger:
                        logger.info(f"Task {task} - Acc: {avg_acc:.4f}")
    
    # 计算结果
    task_end_accs = [all_acc[rounds_per_task * (t + 1) - 1] for t in range(num_tasks)]
    avg_task_acc = sum(task_end_accs) / len(task_end_accs)
    avg_forget = sum(all_forget) / len(all_forget) if all_forget else 0.0
    
    return {
        'coalition_mode': coalition_mode,
        'final_accuracy': all_acc[-1],
        'avg_task_accuracy': avg_task_acc,
        'avg_forgetting': avg_forget,
        'task_end_accs': task_end_accs,
        'all_accuracies': all_acc,
        'all_forgetting': all_forget,
        'coalition_history': coalition_history,
    }


def main():
    """主函数：运行DER++强度消融实验"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = f"results/der_strength_ablation_{timestamp}"
    os.makedirs(result_dir, exist_ok=True)
    
    logger = setup_logger('DERStrengthAblation', os.path.join(result_dir, 'experiment.log'))
    
    logger.info("=" * 80)
    logger.info("DER++强度 vs Coalition机制消融实验")
    logger.info("=" * 80)
    
    # 实验参数
    # buffer_size=0 表示禁用DER++（通过设置der_alpha=der_beta=0）
    buffer_sizes = [0, 50, 100, 200, 500]  # 不同的DER++强度
    seed = 42
    
    # 基础配置（EMNIST-Letters Shuffle模式）
    base_config = Config()
    base_config.dataset = "EMNIST-Letters"
    base_config.data_split_file = "split_files/EMNIST_letters_shuffle_split_cn8_tn6_cet2_cs2_s2571.pkl"
    base_config.num_users = 8
    base_config.num_tasks = 6
    base_config.num_rounds = 60
    base_config.local_epochs = 100
    base_config.batch_size = 64
    base_config.model = "cnn"
    base_config.lr = 1e-4
    base_config.weight_decay = 1e-5
    base_config.seed = seed
    base_config.rounds_per_task = base_config.num_rounds // base_config.num_tasks
    
    # CollEdge 参数
    base_config.algorithm = "CollEdge"
    base_config.sw = 0.1
    base_config.lambda_kd = 0.2
    base_config.lambda_proto_aug = 2.0
    base_config.global_weight = 0.9
    base_config.ema_global = 0.9
    base_config.dcfcl_broadcast = 1
    
    all_results = {}
    
    for buffer_size in buffer_sizes:
        logger.info(f"\n{'#'*80}")
        logger.info(f"Buffer Size: {buffer_size}")
        logger.info(f"{'#'*80}")
        
        all_results[f'buf_{buffer_size}'] = {}
        
        for mode in ['full', 'fedavg']:
            logger.info(f"\n{'='*60}")
            logger.info(f"Coalition Mode: {mode}")
            logger.info(f"{'='*60}")
            
            # 深拷贝配置
            config = copy.deepcopy(base_config)
            
            # 设置buffer_size
            # 注意：即使要禁用DER++，buffer_size也需要>0以避免代码崩溃
            # 通过设置der_alpha=der_beta=0来禁用DER++损失
            if buffer_size == 0:
                config.buffer_size = 10  # 避免代码崩溃
                config.der_alpha = 0.0
                config.der_beta = 0.0
            else:
                config.buffer_size = buffer_size
                config.der_alpha = 0.5
                config.der_beta = 0.5
            
            results = run_experiment(config, coalition_mode=mode, seed=seed, logger=logger)
            all_results[f'buf_{buffer_size}'][mode] = results
            
            logger.info(f"\n结果 (buffer_size={buffer_size}, mode={mode}):")
            logger.info(f"  最终准确率: {results['final_accuracy']*100:.2f}%")
            logger.info(f"  平均任务准确率: {results['avg_task_accuracy']*100:.2f}%")
            logger.info(f"  平均遗忘率: {results['avg_forgetting']*100:.2f}%")
            logger.info(f"  各任务准确率: {[f'{a*100:.1f}%' for a in results['task_end_accs']]}")
    
    # 保存结果
    with open(os.path.join(result_dir, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # 生成报告
    generate_report(all_results, result_dir, logger)
    
    logger.info(f"\n实验完成！结果保存在: {result_dir}")


def generate_report(results, result_dir, logger):
    """生成消融实验报告"""
    
    report = []
    report.append("# DER++强度 vs Coalition机制消融实验报告\n\n")
    report.append(f"**实验时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    report.append("**数据集**: EMNIST-Letters (Shuffle模式，异构数据)\n\n")
    
    report.append("## 实验假设\n\n")
    report.append("DER++回放机制太强（遗忘率=0%）掩盖了Coalition机制的价值。\n")
    report.append("通过降低buffer_size，让遗忘出现，观察Coalition是否能提供更好的保护。\n\n")
    
    report.append("## 实验结果\n\n")
    report.append("| Buffer Size | Mode | 最终准确率 | 平均任务准确率 | 平均遗忘率 |\n")
    report.append("|-------------|------|-----------|---------------|----------|\n")
    
    for buf_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        buffer_size = buf_key.split('_')[1]
        for mode in ['full', 'fedavg']:
            if mode in results[buf_key]:
                r = results[buf_key][mode]
                report.append(f"| {buffer_size} | {mode} | {r['final_accuracy']*100:.2f}% | {r['avg_task_accuracy']*100:.2f}% | {r['avg_forgetting']*100:.2f}% |\n")
    
    report.append("\n## Coalition效果分析 (full - fedavg)\n\n")
    report.append("| Buffer Size | 准确率差异 | 遗忘率差异 | Coalition有效? |\n")
    report.append("|-------------|-----------|----------|---------------|\n")
    
    for buf_key in sorted(results.keys(), key=lambda x: int(x.split('_')[1])):
        buffer_size = buf_key.split('_')[1]
        if 'full' in results[buf_key] and 'fedavg' in results[buf_key]:
            full = results[buf_key]['full']
            fedavg = results[buf_key]['fedavg']
            
            acc_diff = full['final_accuracy'] - fedavg['final_accuracy']
            forget_diff = full['avg_forgetting'] - fedavg['avg_forgetting']
            
            # Coalition有效的标准：准确率更高 或 遗忘率更低
            effective = "✓ 是" if acc_diff > 0.005 or forget_diff < -0.005 else "✗ 否"
            
            report.append(f"| {buffer_size} | {acc_diff*100:+.2f}% | {forget_diff*100:+.2f}% | {effective} |\n")
    
    report.append("\n## 结论\n\n")
    report.append("（根据实验结果自动生成）\n\n")
    
    # 分析趋势
    buf_sizes = sorted([int(k.split('_')[1]) for k in results.keys()])
    acc_diffs = []
    forget_diffs = []
    
    for buf_size in buf_sizes:
        buf_key = f'buf_{buf_size}'
        if 'full' in results[buf_key] and 'fedavg' in results[buf_key]:
            acc_diff = results[buf_key]['full']['final_accuracy'] - results[buf_key]['fedavg']['final_accuracy']
            forget_diff = results[buf_key]['full']['avg_forgetting'] - results[buf_key]['fedavg']['avg_forgetting']
            acc_diffs.append(acc_diff)
            forget_diffs.append(forget_diff)
    
    if acc_diffs:
        # 检查趋势：buffer_size越小，Coalition优势越大？
        if acc_diffs[0] > acc_diffs[-1]:
            report.append("- **假设验证成功**: 随着buffer_size减小（DER++减弱），Coalition机制的优势增大\n")
        else:
            report.append("- **假设未验证**: Coalition优势未随DER++减弱而增加\n")
        
        report.append(f"- 最小buffer (buf={buf_sizes[0]}): full vs fedavg = {acc_diffs[0]*100:+.2f}%\n")
        report.append(f"- 最大buffer (buf={buf_sizes[-1]}): full vs fedavg = {acc_diffs[-1]*100:+.2f}%\n")
    
    with open(os.path.join(result_dir, 'ABLATION_REPORT.md'), 'w') as f:
        f.writelines(report)
    
    logger.info("\n" + "="*60)
    logger.info("实验报告已生成: ABLATION_REPORT.md")
    logger.info("="*60)


if __name__ == '__main__':
    main()
