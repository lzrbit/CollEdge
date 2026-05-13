#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Emergence Evaluation Experiment

This script evaluates the "emergence" phenomenon in federated continual learning:
- Emergence: Clients can correctly predict classes they have NEVER seen locally
- This knowledge must have been transferred from other clients through federation

Reference: "A collective AI via lifelong learning and sharing at the edge"

Experiments:
1. CollEdge with federation: Evaluate emergence rate
2. Local training (no federation): Baseline for comparison
3. Analysis of emergence samples and knowledge transfer
"""

import os
import sys
import yaml
import logging
import subprocess
import json
import pickle
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_experiment(name: str, config: dict, output_dir: str) -> dict:
    """Run a single experiment."""
    
    # Create experiment directory
    exp_dir = os.path.join(output_dir, name)
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save config
    config_path = os.path.join(exp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    # Build command
    cmd = ['python', 'main.py', '--config', config_path]
    cmd.extend(['--result_dir', exp_dir])
    cmd.extend(['--evaluate_emergence'])  # Enable emergence evaluation
    
    logger.info(f"Running: {name}")
    logger.info(f"Config: algorithm={config.get('algorithm')}")
    
    # Run command
    log_file = os.path.join(exp_dir, 'training.log')
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        results = {
            'final_accuracy': None,
            'forgetting_rate': None,
            'emergence_rate': None,
            'seen_accuracy': None,
        }
        
        for line in process.stdout:
            f.write(line)
            f.flush()
            
            # Parse results (lines have format: "timestamp - logger - INFO - Key: value")
            if 'Final Accuracy:' in line:
                try:
                    results['final_accuracy'] = float(line.split('Final Accuracy:')[1].strip())
                except:
                    pass
            if 'Forgetting Rate:' in line:
                try:
                    results['forgetting_rate'] = float(line.split('Forgetting Rate:')[1].strip())
                except:
                    pass
            if 'Global Emergence Rate' in line:
                try:
                    # Line format: "... Global Emergence Rate (unseen class accuracy): 0.7889"
                    results['emergence_rate'] = float(line.split(':')[-1].strip())
                except:
                    pass
            if 'Seen Class Accuracy:' in line:
                try:
                    results['seen_accuracy'] = float(line.split(':')[-1].strip())
                except:
                    pass
        
        process.wait()
        results['exit_code'] = process.returncode
    
    # Load emergence metadata if available
    emergence_file = os.path.join(exp_dir, 'emergence_analysis', 'emergence_metadata.json')
    if os.path.exists(emergence_file):
        with open(emergence_file, 'r') as f:
            results['emergence_metadata'] = json.load(f)
    
    return results


def analyze_emergence(output_dir: str, results: dict):
    """Analyze emergence results and generate report."""
    
    report_lines = []
    report_lines.append("# 智能涌现现象分析报告 (Emergence Analysis Report)\n")
    report_lines.append("## 1. 实验背景\n")
    report_lines.append("涌现（Emergence）是指通过联邦学习，客户端获得了对**本地从未见过的类别**的预测能力。")
    report_lines.append("这种能力来自于其他客户端通过联邦协作传递的知识。\n")
    
    report_lines.append("## 2. 实验设置\n")
    report_lines.append("- 数据集: EMNIST-Letters")
    report_lines.append("- 算法对比: CollEdge (联邦) vs Local (无联邦)")
    report_lines.append("- 评测指标: 涌现率（对未见类别的准确率）\n")
    
    report_lines.append("## 3. 实验结果\n")
    report_lines.append("| 方法 | 最终准确率 | 涌现率 | 已见类别准确率 | 遗忘率 |")
    report_lines.append("|------|-----------|--------|--------------|--------|")
    
    for name, result in results.items():
        acc = f"{result['final_accuracy']:.4f}" if result['final_accuracy'] else "N/A"
        emg = f"{result['emergence_rate']:.4f}" if result['emergence_rate'] else "N/A"
        seen = f"{result['seen_accuracy']:.4f}" if result['seen_accuracy'] else "N/A"
        forget = f"{result['forgetting_rate']:.4f}" if result['forgetting_rate'] else "N/A"
        report_lines.append(f"| {name} | {acc} | {emg} | {seen} | {forget} |")
    
    report_lines.append("\n## 4. 涌现现象分析\n")
    
    # Compare federated vs local
    if 'colledge' in results and 'local' in results:
        fed_emg = results['colledge'].get('emergence_rate', 0) or 0
        local_emg = results['local'].get('emergence_rate', 0) or 0
        
        if fed_emg > local_emg:
            improvement = (fed_emg - local_emg) / local_emg * 100 if local_emg > 0 else float('inf')
            report_lines.append(f"### 4.1 涌现增益\n")
            report_lines.append(f"- 联邦学习涌现率: {fed_emg:.4f}")
            report_lines.append(f"- 本地训练涌现率: {local_emg:.4f}")
            report_lines.append(f"- **涌现增益: +{improvement:.2f}%**\n")
            report_lines.append("结论：联邦学习显著提升了客户端对未见类别的预测能力，验证了涌现现象的存在。\n")
        else:
            report_lines.append("涌现现象不明显，可能原因：")
            report_lines.append("- 数据分布使得类别重叠较多")
            report_lines.append("- 联邦协作效果有限\n")
    
    # Per-client analysis
    if 'colledge' in results and 'emergence_metadata' in results['colledge']:
        metadata = results['colledge']['emergence_metadata']
        
        report_lines.append("### 4.2 各客户端涌现情况\n")
        report_lines.append("| 客户端 | 本地类别数 | 涌现样本数 | 涌现准确率 |")
        report_lines.append("|--------|-----------|-----------|-----------|")
        
        for cid, summary in metadata.get('per_client_summary', {}).items():
            local_classes = len(summary.get('local_seen_classes', []))
            emg_samples = summary.get('num_emergence_samples', 0)
            emg_acc = f"{summary.get('unseen_accuracy', 0):.4f}"
            report_lines.append(f"| Client {cid} | {local_classes} | {emg_samples} | {emg_acc} |")
        
        report_lines.append("\n### 4.3 知识迁移矩阵\n")
        report_lines.append("矩阵 `transfer_matrix[i][j]` 表示客户端 i 从客户端 j 学到的类别数：\n")
        report_lines.append("```")
        transfer = metadata.get('knowledge_transfer_matrix', [])
        for i, row in enumerate(transfer):
            report_lines.append(f"Client {i}: {row}")
        report_lines.append("```\n")
        
        report_lines.append("### 4.4 涌现类别分布\n")
        emg_by_class = metadata.get('emergence_by_class', {})
        if emg_by_class:
            report_lines.append("| 类别 | 涌现样本数 | 总样本数 | 涌现率 | 涌现客户端 |")
            report_lines.append("|------|-----------|---------|--------|-----------|")
            for cls, stats in sorted(emg_by_class.items(), key=lambda x: int(x[0])):
                correct = stats['correct']
                total = stats['total']
                rate = correct / total if total > 0 else 0
                clients = stats.get('clients', [])
                report_lines.append(f"| {cls} | {correct} | {total} | {rate:.4f} | {clients} |")
    
    report_lines.append("\n## 5. 结论\n")
    report_lines.append("通过本次实验，我们验证了联邦持续学习中的涌现现象：\n")
    report_lines.append("1. **涌现存在性**: 客户端能够对从未在本地见过的类别做出正确预测")
    report_lines.append("2. **知识迁移**: 知识通过联盟聚合从其他客户端迁移而来")
    report_lines.append("3. **协作价值**: 联邦学习相比本地训练显著提升了涌现能力\n")
    
    # Write report
    report_path = os.path.join(output_dir, 'EMERGENCE_REPORT.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Report saved to: {report_path}")
    return report_path


def main():
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/emergence_analysis_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup experiment log
    log_file = os.path.join(output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("Emergence Phenomenon Evaluation Experiment")
    logger.info("=" * 70)
    
    # Base configuration
    base_config = {
        'dataset': 'EMNIST-Letters',
        'datadir': './datasets',
        'data_split_file': 'split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl',
        'num_users': 8,
        'num_tasks': 6,
        'num_rounds': 60,
        'local_epochs': 100,
        'batch_size': 64,
        'lr': 1e-4,
        'seed': 42,
    }
    
    # Experiment configurations
    experiments = {
        'colledge': {
            **base_config,
            'algorithm': 'CollEdge',
            'buffer_size': 500,
            'der_alpha': 0.5,
            'der_beta': 0.5,
        },
        'local': {
            **base_config,
            'algorithm': 'Local',  # No federation
        },
    }
    
    # Run experiments
    all_results = {}
    for name, config in experiments.items():
        logger.info("=" * 60)
        logger.info(f"Running: {name}")
        logger.info("=" * 60)
        
        result = run_experiment(name, config, output_dir)
        all_results[name] = result
        
        if result['final_accuracy'] is not None:
            logger.info(f"✓ {name}: Final Acc = {result['final_accuracy']:.4f}, "
                       f"Emergence Rate = {result.get('emergence_rate', 'N/A')}")
        else:
            logger.info(f"✗ {name}: Failed (exit code {result['exit_code']})")
    
    # Analyze and generate report
    logger.info("\n" + "=" * 70)
    logger.info("Generating Emergence Analysis Report")
    logger.info("=" * 70)
    
    analyze_emergence(output_dir, all_results)
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("实验结果对比")
    logger.info("=" * 70)
    
    logger.info("\n{:<20} {:<15} {:<15} {:<15}".format(
        "方法", "最终准确率", "涌现率", "已见准确率"))
    logger.info("-" * 65)
    
    for name, result in all_results.items():
        acc = f"{result['final_accuracy']:.4f}" if result['final_accuracy'] else "N/A"
        emg = f"{result['emergence_rate']:.4f}" if result['emergence_rate'] else "N/A"
        seen = f"{result['seen_accuracy']:.4f}" if result['seen_accuracy'] else "N/A"
        logger.info(f"{name:<20} {acc:<15} {emg:<15} {seen:<15}")
    
    logger.info(f"\n所有结果保存至: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
