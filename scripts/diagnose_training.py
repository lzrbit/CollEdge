#!/usr/bin/env python
"""Quick diagnostic training to verify algorithms work correctly."""
import sys
import os
import logging
import torch
import numpy as np

# Setup path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed

def run_diagnostic(algorithm, num_rounds=12, num_tasks=2):
    """Run a short diagnostic training and report results."""
    setup_seed(42)
    
    config = Config()
    config.algorithm = algorithm
    config.num_rounds = num_rounds
    config.num_tasks = num_tasks
    config.local_epochs = 50  # Shorter for speed
    config.rounds_per_task = num_rounds // num_tasks
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Suppress verbose logging
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Client').setLevel(logging.WARNING)
    
    print(f"\n{'='*60}")
    print(f"Algorithm: {algorithm}")
    print(f"lr={config.lr}, wd={config.weight_decay}, epochs={config.local_epochs}")
    print(f"rounds={num_rounds}, tasks={num_tasks}, device={device}")
    if algorithm == 'DCFCL':
        print(f"lambda_kd={config.lambda_kd}, sw={config.sw}, global_weight={config.global_weight}")
    elif algorithm == 'FedLwF':
        print(f"alpha={config.alpha}")
    print(f"{'='*60}")
    
    server = DCFCLServer(config, device)
    
    # Train with per-round reporting
    rounds_per_task = config.rounds_per_task
    all_accs = []
    
    for task in range(num_tasks):
        print(f"\n--- Task {task} ---")
        
        if task > 0:
            server._update_clients_for_new_task(task)
        server._update_available_labels()
        
        for round_in_task in range(rounds_per_task):
            global_round = task * rounds_per_task + round_in_task
            
            # Distribute model (except Local, and except first round for DCFCL)
            if config.algorithm not in ['Local', 'DCFCL'] and global_round == 0:
                server._broadcast_parameters()
            elif config.algorithm == 'DCFCL' and global_round == 0:
                server._broadcast_parameters()
            
            # Local training
            train_results = server._local_training(global_round, task)
            
            # Server aggregation
            server._server_aggregation(global_round, task, train_results)
            
            # Evaluate
            accs, avg_acc, num_samples = server._evaluate()
            all_accs.append(avg_acc)
            
            # Track per-task accuracy at end of task
            if round_in_task == rounds_per_task - 1:
                server.all_accs.append(accs)
            
            print(f"  Round {global_round}: avg_acc={avg_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"RESULT [{algorithm}]: Final accuracy = {all_accs[-1]:.4f}")
    print(f"Accuracy progression: {[f'{a:.3f}' for a in all_accs]}")
    print(f"{'='*60}")
    
    return all_accs[-1]

if __name__ == '__main__':
    algorithms = sys.argv[1:] if len(sys.argv) > 1 else ['FedAvg', 'DCFCL']
    
    results = {}
    for algo in algorithms:
        try:
            acc = run_diagnostic(algo, num_rounds=12, num_tasks=2)
            results[algo] = acc
        except Exception as e:
            print(f"ERROR [{algo}]: {e}")
            import traceback
            traceback.print_exc()
            results[algo] = -1
    
    print(f"\n\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for algo, acc in results.items():
        print(f"  {algo:15s}: {acc:.4f}")
