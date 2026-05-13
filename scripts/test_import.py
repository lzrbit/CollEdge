#!/usr/bin/env python3
"""Test script to verify the refactored codebase."""

import sys
print(f"Python version: {sys.version}")

# Test imports
print("\n=== Testing Imports ===")

try:
    from core.config import Config
    print("✓ core.config imported successfully")
except Exception as e:
    print(f"✗ core.config failed: {e}")

try:
    from core.models import create_model, SimpleCNN, ResNet18CBAM
    print("✓ core.models imported successfully")
except Exception as e:
    print(f"✗ core.models failed: {e}")

try:
    from core.client import Client
    print("✓ core.client imported successfully")
except Exception as e:
    print(f"✗ core.client failed: {e}")

try:
    from core.server import DCFCLServer
    print("✓ core.server imported successfully")
except Exception as e:
    print(f"✗ core.server failed: {e}")

try:
    from core.optimizers import ScaffoldOptimizer, PerAvgOptimizer, pFedMeOptimizer
    print("✓ core.optimizers imported successfully")
except Exception as e:
    print(f"✗ core.optimizers failed: {e}")

try:
    from utils.data_loader import get_dataset, read_user_data
    print("✓ utils.data_loader imported successfully")
except Exception as e:
    print(f"✗ utils.data_loader failed: {e}")

try:
    from utils.helpers import setup_seed, setup_logging
    print("✓ utils.helpers imported successfully")
except Exception as e:
    print(f"✗ utils.helpers failed: {e}")

# Test model creation
print("\n=== Testing Model Creation ===")
import torch

try:
    # Test SimpleCNN directly
    from core.models import SimpleCNN
    model = SimpleCNN(image_size=28, in_channels=1, num_classes=26, feature_dim=256)
    x = torch.randn(2, 1, 28, 28)
    probs, features, logits = model(x)
    print(f"✓ SimpleCNN: input {x.shape} -> features {features.shape}, logits {logits.shape}")
except Exception as e:
    print(f"✗ SimpleCNN failed: {e}")

try:
    # Test ResNet18CBAM directly
    from core.models import ResNet18CBAM
    model = ResNet18CBAM(num_classes=100, feature_dim=512)
    x = torch.randn(2, 3, 32, 32)
    probs, features, logits = model(x)
    print(f"✓ ResNet18CBAM: input {x.shape} -> features {features.shape}, logits {logits.shape}")
except Exception as e:
    print(f"✗ ResNet18CBAM failed: {e}")

try:
    # Test DCFCLModel via create_model
    from core.models import create_model
    from core.config import Config
    config = Config(dataset='EMNIST-Letters')
    model = create_model(config)
    x = torch.randn(2, 1, 28, 28)
    probs, features, logits = model(x)
    print(f"✓ DCFCLModel: input {x.shape} -> features {features.shape}, logits {logits.shape}")
except Exception as e:
    print(f"✗ DCFCLModel failed: {e}")

# Test Config
print("\n=== Testing Config ===")
try:
    from core.config import Config
    config = Config()
    print(f"✓ Default config created: algorithm={config.algorithm}, dataset={config.dataset}")
except Exception as e:
    print(f"✗ Config creation failed: {e}")

# Test YAML config loading
print("\n=== Testing YAML Config Loading ===")
try:
    import yaml
    import os
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'dcfcl_emnist.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            yaml_config = yaml.safe_load(f)
        config = Config(**yaml_config)
        print(f"✓ YAML config loaded: algorithm={config.algorithm}, dataset={config.dataset}")
    else:
        print(f"! Config file not found: {config_path}")
except Exception as e:
    print(f"✗ YAML config loading failed: {e}")

print("\n=== All Basic Tests Complete ===")
