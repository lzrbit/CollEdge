#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Coalition Mask Ablation Experiment

This script tests the coalition mask functionality and compares performance:
1. baseline: No coalition mask (all clients can cooperate)
2. group_mask: Group-based mask (clients in different groups cannot cooperate)
3. random_mask: Random forbidden pairs

The goal is to verify that:
1. The coalition mask works correctly
2. Understand how restricting coalitions affects performance
"""

import os
import sys
import yaml
import logging
import subprocess
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
    
    # Override result_dir
    cmd.extend(['--result_dir', exp_dir])
    
    logger.info(f"Running: {name}")
    logger.info(f"Config: use_coalition_mask={config.get('use_coalition_mask', False)}, "
                f"mask_type={config.get('coalition_mask_type', 'custom')}")
    
    # Run command
    log_file = os.path.join(exp_dir, 'training.log')
    with open(log_file, 'w') as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        final_accuracy = None
        forgetting_rate = None
        
        for line in process.stdout:
            f.write(line)
            f.flush()
            
            # Parse results
            if 'Final Accuracy:' in line:
                try:
                    final_accuracy = float(line.split('Final Accuracy:')[1].strip())
                except:
                    pass
            if 'Forgetting Rate:' in line:
                try:
                    forgetting_rate = float(line.split('Forgetting Rate:')[1].strip())
                except:
                    pass
        
        process.wait()
    
    return {
        'name': name,
        'final_accuracy': final_accuracy,
        'forgetting_rate': forgetting_rate,
        'exit_code': process.returncode
    }


def main():
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = f'results/coalition_mask_test_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup experiment log
    log_file = os.path.join(output_dir, 'experiment.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("=" * 70)
    logger.info("Coalition Mask Ablation Experiment")
    logger.info("=" * 70)
    
    # Base configuration
    base_config = {
        'dataset': 'EMNIST-Letters',
        'datadir': './datasets',
        'data_split_file': 'split_files/EMNIST_letters_split_cn8_tn6_cet2_cs2_s2571.pkl',
        'algorithm': 'CollEdge',
        'num_users': 8,
        'num_tasks': 6,
        'num_rounds': 60,
        'local_epochs': 100,
        'batch_size': 64,
        'lr': 1e-4,
        'buffer_size': 100,  # Lower to allow some forgetting
        'der_alpha': 0.5,
        'der_beta': 0.5,
        'seed': 42,
    }
    
    # Experiment configurations
    experiments = {
        'baseline': {
            **base_config,
            'use_coalition_mask': False,
        },
        'group_mask_2': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'group',
            'num_client_groups': 2,  # Split 8 clients into 2 groups of 4
        },
        'group_mask_4': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'group',
            'num_client_groups': 4,  # Split 8 clients into 4 groups of 2
        },
        'random_mask_20': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'random',
            'coalition_mask_density': 0.2,  # 20% pairs forbidden
        },
        'random_mask_50': {
            **base_config,
            'use_coalition_mask': True,
            'coalition_mask_type': 'random',
            'coalition_mask_density': 0.5,  # 50% pairs forbidden
        },
    }
    
    # Run experiments
    results = []
    for name, config in experiments.items():
        logger.info("=" * 60)
        logger.info(f"Running: {name}")
        logger.info("=" * 60)
        
        result = run_experiment(name, config, output_dir)
        results.append(result)
        
        if result['final_accuracy'] is not None:
            logger.info(f"✓ {name}: Final Acc = {result['final_accuracy']:.4f}")
        else:
            logger.info(f"✗ {name}: Failed (exit code {result['exit_code']})")
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("实验结果对比")
    logger.info("=" * 70)
    
    logger.info("\n{:<25} {:<15} {:<15}".format("方法", "最终准确率", "遗忘率"))
    logger.info("-" * 55)
    
    baseline_acc = None
    for r in results:
        if r['final_accuracy'] is not None:
            acc_str = f"{r['final_accuracy']:.4f}"
            forget_str = f"{r['forgetting_rate']:.4f}" if r['forgetting_rate'] else "N/A"
            logger.info(f"{r['name']:<25} {acc_str:<15} {forget_str:<15}")
            
            if r['name'] == 'baseline':
                baseline_acc = r['final_accuracy']
    
    # Compare with baseline
    if baseline_acc is not None:
        logger.info("-" * 55)
        logger.info("\n与 Baseline 对比:")
        for r in results:
            if r['name'] != 'baseline' and r['final_accuracy'] is not None:
                diff = r['final_accuracy'] - baseline_acc
                pct = (diff / baseline_acc) * 100
                sign = '+' if diff >= 0 else ''
                logger.info(f"  {r['name']}: {sign}{diff:.4f} ({sign}{pct:.2f}%)")
    
    # Generate report
    report_path = os.path.join(output_dir, 'COALITION_MASK_REPORT.md')
    with open(report_path, 'w') as f:
        f.write("# Coalition Mask 消融实验报告\n\n")
        f.write("## 实验设置\n")
        f.write("- 数据集: EMNIST-Letters\n")
        f.write("- 算法: CollEdge\n")
        f.write("- buffer_size: 100\n")
        f.write("- 客户端数: 8\n")
        f.write("- 任务数: 6\n\n")
        
        f.write("## 实验结果\n\n")
        f.write("| 方法 | 最终准确率 | 遗忘率 | 说明 |\n")
        f.write("|------|-----------|--------|------|\n")
        
        descriptions = {
            'baseline': '无约束，所有客户端可合作',
            'group_mask_2': '2组约束：客户端0-3为组1，4-7为组2',
            'group_mask_4': '4组约束：每2个客户端一组',
            'random_mask_20': '20%随机禁止对',
            'random_mask_50': '50%随机禁止对',
        }
        
        for r in results:
            if r['final_accuracy'] is not None:
                acc = f"{r['final_accuracy']:.4f}"
                forget = f"{r['forgetting_rate']:.4f}" if r['forgetting_rate'] else "N/A"
                desc = descriptions.get(r['name'], '')
                f.write(f"| {r['name']} | {acc} | {forget} | {desc} |\n")
        
        f.write("\n## 结论\n\n")
        if baseline_acc is not None:
            worst_result = min([r for r in results if r['final_accuracy']], 
                              key=lambda x: x['final_accuracy'] if x['final_accuracy'] else float('inf'))
            best_result = max([r for r in results if r['final_accuracy']], 
                             key=lambda x: x['final_accuracy'] if x['final_accuracy'] else 0)
            
            f.write(f"- 最佳方法: {best_result['name']} ({best_result['final_accuracy']:.4f})\n")
            f.write(f"- 最差方法: {worst_result['name']} ({worst_result['final_accuracy']:.4f})\n")
            
            # Analyze impact
            f.write("\n### 约束对性能的影响\n\n")
            for r in results:
                if r['name'] != 'baseline' and r['final_accuracy'] is not None:
                    diff = r['final_accuracy'] - baseline_acc
                    if diff < -0.01:
                        f.write(f"- **{r['name']}**: 性能下降 {abs(diff):.4f}，约束限制了有效协作\n")
                    elif diff > 0.01:
                        f.write(f"- **{r['name']}**: 性能提升 {diff:.4f}，约束可能避免了负迁移\n")
                    else:
                        f.write(f"- **{r['name']}**: 性能基本不变，约束影响较小\n")
    
    logger.info(f"\n报告已保存至: {report_path}")
    logger.info(f"所有结果保存至: {output_dir}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
