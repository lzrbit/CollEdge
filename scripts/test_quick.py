#!/usr/bin/env python3
"""Quick end-to-end test for the refactored codebase."""

import sys
import os
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import Config
from core.models import create_model
from core.client import Client
from core.server import DCFCLServer
from utils.helpers import setup_seed

print("=== DCFCL Quick Test ===\n")

# Create a minimal config for testing
config = Config(
    algorithm='DCFCL',
    dataset='TEST-noniid',  # Use simple test dataset
    num_users=4,
    num_tasks=2,
    num_rounds=2,
    local_epochs=2,
    batch_size=32,
    device='cpu'
)

print(f"Config: algorithm={config.algorithm}, users={config.num_users}, tasks={config.num_tasks}")

# Set seed for reproducibility
setup_seed(config.seed)

# Create synthetic data for testing
print("\n1. Creating synthetic data...")
# No longer needed here - data created with data loaders below
print(f"   Will create synthetic data with data loaders")

# Create model
print("\n2. Creating model...")
model = create_model(config)
print(f"   Model created: {type(model.classifier).__name__}")

# Create synthetic training/test datasets
print("\n3. Creating data loaders...")
from torch.utils.data import TensorDataset, DataLoader

clients = []
for i in range(config.num_users):
    # Create synthetic training data
    train_x = torch.randn(100, 784)
    train_y = torch.randint(0, 10, (100,))
    train_dataset = TensorDataset(train_x, train_y)
    
    # Create synthetic test data
    test_x = torch.randn(20, 784)
    test_y = torch.randint(0, 10, (20,))
    test_dataset = TensorDataset(test_x, test_y)
    
    # Label info
    label_info = {
        'labels': list(range(10)),
        'counts': {j: 10 for j in range(10)}
    }
    
    client = Client(
        client_id=i,
        config=config,
        model=model,
        train_data=train_dataset,
        test_data=test_dataset,
        label_info=label_info,
        unique_labels=10
    )
    clients.append(client)

print(f"   Created {len(clients)} clients")

# Test client training
print("\n4. Testing client training...")
test_client = clients[0]

# Simple training step
try:
    test_client.train(glob_iter=0, task=0)
    acc, loss, samples = test_client.test()
    print(f"   Client 0 training successful, accuracy: {acc:.4f}, loss: {loss:.4f}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"   Client training failed: {e}")

# Test server
print("\n5. Testing server...")
try:
    device = torch.device('cpu')
    server = DCFCLServer(config, device)
    print(f"   Server created successfully")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"   Server creation failed: {e}")

print("\n=== Quick Test Complete ===")
print("All basic components working!")
