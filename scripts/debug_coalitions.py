#!/usr/bin/env python
"""Debug DCFCL coalition formation to understand why it underperforms FedAvg."""
import sys, os, logging, torch, numpy as np, copy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core.config import Config
from core.server import DCFCLServer
from utils.helpers import setup_seed

def run_debug(num_rounds=12, num_tasks=2):
    setup_seed(42)
    config = Config()
    config.algorithm = 'DCFCL'
    config.num_rounds = num_rounds
    config.num_tasks = num_tasks
    config.local_epochs = 50
    config.rounds_per_task = num_rounds // num_tasks
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.getLogger('DCFCL').setLevel(logging.WARNING)
    logging.getLogger('DCFCL.Server').setLevel(logging.WARNING)
    
    server = DCFCLServer(config, device)
    rounds_per_task = config.rounds_per_task
    
    for task in range(num_tasks):
        if task > 0:
            server._update_clients_for_new_task(task)
        server._update_available_labels()
        
        for round_in_task in range(rounds_per_task):
            global_round = task * rounds_per_task + round_in_task
            
            if config.algorithm != 'Local' and global_round == 0:
                server._broadcast_parameters()
            
            # Local training
            train_results = server._local_training(global_round, task)
            
            # === Debug: examine client states before aggregation ===
            print(f"\n--- Round {global_round} (T{task}R{round_in_task}) ---")
            print(f"Client l2_norms: {[f'{c.l2_norm:.4f}' if c.l2_norm else 'None' for c in server.clients]}")
            
            # Aggregate
            algorithm = config.algorithm
            w_global = server._aggregate_weights(train_results['w_locals'])
            
            # EMA
            old_model = copy.deepcopy(server.model)
            server.model.load_state_dict(w_global)
            gw = config.global_weight
            for old_p, new_p in zip(old_model.parameters(), server.model.parameters()):
                new_p.data = old_p.data * (1 - gw) + new_p.data.clone() * gw
            
            # Prototype aggregation
            if config.algorithm == 'DCFCL':
                server.proto_global = server._aggregate_prototypes(train_results['proto_locals'])
                server.radius_global = server._aggregate_radius(train_results['radius_locals'])
            
            # Similarity matrix
            server.similarity_matrix = server._compute_similarity_matrix()
            print(f"Similarity matrix:\n{np.array2string(server.similarity_matrix, precision=3, suppress_small=True)}")
            
            # Update pay table
            server._update_pay_table()
            
            # Form coalitions
            if global_round == 0:
                server._form_coalition_initial()
            else:
                server._form_coalition_dynamic()
            
            print(f"Coalitions: {server.unions}")
            
            # Check: what would accuracy be WITHOUT coalition (just FedAvg broadcast)?
            # Save current client models
            saved_models = [copy.deepcopy(c.model.state_dict()) for c in server.clients]
            
            # Test 1: distribute coalition models
            server._aggregate_coalitions()
            server._distribute_coalition_models()
            
            accs_coalition, avg_acc_coalition, _ = server._evaluate()
            
            # Test 2: restore and distribute FedAvg model (the EMA-blended server model)
            for c in server.clients:
                c.set_parameters(server.model)
            
            accs_fedavg, avg_acc_fedavg, _ = server._evaluate()
            
            # Test 3: no distribution (keep local models)
            for i, c in enumerate(server.clients):
                c.model.load_state_dict(saved_models[i])
            
            accs_local, avg_acc_local, _ = server._evaluate()
            
            print(f"Accuracy comparison:")
            print(f"  Local (no agg):  {avg_acc_local:.4f}")
            print(f"  FedAvg (global): {avg_acc_fedavg:.4f}")
            print(f"  Coalition:       {avg_acc_coalition:.4f}")
            
            # Restore coalition distribution for next round
            server._distribute_coalition_models()
            
            if round_in_task == rounds_per_task - 1:
                server.all_accs.append(accs_coalition)

if __name__ == '__main__':
    run_debug(num_rounds=12, num_tasks=2)
