#!/usr/bin/env python3
"""
Directed Collaboration 对比实验

目的: 验证有向协作机制相比标准对称联盟聚合是否有效果

实验设计:
1. baseline: 标准对称联盟聚合 (directed_collaboration=False)
2. directed_gradient: 基于梯度的有向协作

使用较小的 buffer_size (100) 来允许一些遗忘发生，
这样 directed collaboration 的个性化优势才能体现。
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import yaml
import json
import re
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_experiment(config_overrides: dict, exp_name: str, base_config: str, result_dir: Path):
    """Run a single experiment with config overrides."""
    # Load base config
    with open(base_config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply overrides
    config.update(config_overrides)
    
    # Set result directory
    config['result_dir'] = str(result_dir / exp_name)
    
    # Create temporary config file
    temp_config_path = result_dir / f'{exp_name}_config.yaml'
    with open(temp_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"{'='*60}")
    logger.info(f"Running: {exp_name}")
    logger.info(f"Config: directed_collaboration={config.get('directed_collaboration', False)}, buffer_size={config.get('buffer_size', 500)}")
    logger.info(f"{'='*60}")
    
    # Run the experiment
    result = subprocess.run(
        ['python', 'main.py', '--config', str(temp_config_path)],
        capture_output=True,
        text=True,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    
    if result.returncode != 0:
        logger.error(f"Error running {exp_name}:")
        logger.error(result.stderr[-2000:] if len(result.stderr) > 2000 else result.stderr)
        return None
    
    # Parse results from output
    output = result.stdout + result.stderr
    metrics = parse_metrics(output)
    metrics['exp_name'] = exp_name
    metrics['config'] = config_overrides
    
    return metrics


def parse_metrics(output: str) -> dict:
    """Parse metrics from experiment output."""
    metrics = {}
    
    # Find final accuracy
    final_acc_match = re.search(r'Final.*[Aa]ccuracy[:\s]+(\d+\.?\d*)', output)
    if final_acc_match:
        metrics['final_accuracy'] = float(final_acc_match.group(1))
    
    # Find average accuracy
    avg_acc_match = re.search(r'[Aa]verage.*[Aa]ccuracy[:\s]+(\d+\.?\d*)', output)
    if avg_acc_match:
        metrics['avg_accuracy'] = float(avg_acc_match.group(1))
    
    # Find per-task accuracies
    task_acc_match = re.search(r'Per-task accuracy.*?:\s*\[([\d.,\s]+)\]', output)
    if task_acc_match:
        try:
            accs = [float(x.strip()) for x in task_acc_match.group(1).split(',')]
            metrics['task_accuracies'] = accs
        except:
            pass
    
    # Find forgetting
    forget_match = re.search(r'[Ff]orget(?:ting)?[:\s]+(\d+\.?\d*)', output)
    if forget_match:
        metrics['forgetting'] = float(forget_match.group(1))
    
    # Find last few accuracy values from log
    acc_values = re.findall(r'Average accuracy:\s*(\d+\.?\d*)', output)
    if acc_values:
        metrics['final_round_accuracy'] = float(acc_values[-1])
        metrics['all_round_accuracies'] = [float(x) for x in acc_values]
    
    return metrics


def main():
    """Run directed collaboration comparison experiment."""
    logger.info("="*70)
    logger.info("DIRECTED COLLABORATION 对比实验")
    logger.info("="*70)
    
    # Create result directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_dir = Path(f'./results/directed_comparison_{timestamp}')
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup file logging
    fh = logging.FileHandler(result_dir / 'experiment.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    base_config = './configs/dcfcl_emnist.yaml'
    
    # Define experiments - use buffer_size=100 to allow some forgetting
    experiments = [
        {
            'name': 'baseline',
            'overrides': {
                'algorithm': 'CollEdge',
                'directed_collaboration': False,
                'buffer_size': 100,
            }
        },
        {
            'name': 'directed_gradient',
            'overrides': {
                'algorithm': 'CollEdge',
                'directed_collaboration': True,
                'directed_mode': 'gradient',
                'directed_threshold': 0.0,
                'directed_temperature': 1.0,
                'directed_self_weight': 0.5,
                'buffer_size': 100,
            }
        },
    ]
    
    # Run experiments
    results = []
    
    for exp in experiments:
        result = run_experiment(
            exp['overrides'], 
            exp['name'], 
            base_config,
            result_dir
        )
        if result:
            results.append(result)
            logger.info(f"✓ {exp['name']}: Final Acc = {result.get('final_round_accuracy', 'N/A')}")
        else:
            logger.error(f"✗ {exp['name']}: FAILED")
    
    # Save results
    with open(result_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("实验结果对比")
    logger.info("="*70)
    
    baseline_result = next((r for r in results if r['exp_name'] == 'baseline'), None)
    directed_result = next((r for r in results if r['exp_name'] == 'directed_gradient'), None)
    
    if baseline_result and directed_result:
        baseline_acc = baseline_result.get('final_round_accuracy', 0)
        directed_acc = directed_result.get('final_round_accuracy', 0)
        improvement = directed_acc - baseline_acc
        
        logger.info(f"\n{'方法':<25} {'最终准确率':<15}")
        logger.info("-" * 40)
        logger.info(f"{'Baseline (对称联盟)':<25} {baseline_acc:.4f}")
        logger.info(f"{'Directed (有向协作)':<25} {directed_acc:.4f}")
        logger.info("-" * 40)
        logger.info(f"{'改进':<25} {improvement:+.4f} ({improvement*100:+.2f}%)")
        
        # Write summary report
        report = f"""# Directed Collaboration 对比实验报告

## 实验设置
- 数据集: EMNIST-Letters
- 算法: CollEdge
- buffer_size: 100 (降低以允许遗忘)
- 客户端数: 8
- 任务数: 6

## 实验结果

| 方法 | 最终准确率 |
|------|-----------|
| Baseline (对称联盟) | {baseline_acc:.4f} |
| Directed (有向协作) | {directed_acc:.4f} |

## 结论

改进幅度: **{improvement:+.4f}** ({improvement*100:+.2f}%)

{"有向协作机制有效，提升了模型准确率。" if improvement > 0 else "有向协作机制未能提升准确率，需要进一步调整参数。"}

## 详细结果

### Baseline
- 配置: directed_collaboration=False
- 最终准确率: {baseline_acc:.4f}

### Directed Gradient
- 配置: directed_collaboration=True, mode=gradient
- 最终准确率: {directed_acc:.4f}
"""
        with open(result_dir / 'COMPARISON_REPORT.md', 'w') as f:
            f.write(report)
        
        logger.info(f"\n报告已保存至: {result_dir / 'COMPARISON_REPORT.md'}")
    
    logger.info("\n" + "="*70)
    logger.info(f"所有结果保存至: {result_dir}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
