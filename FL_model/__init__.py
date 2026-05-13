#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FL_model: 模块化联邦学习算法实现。

每种算法对应一个独立的 Python 文件，继承自 BaseClient。
使用 create_client() 工厂函数根据 config.algorithm 创建对应客户端。
"""

from .base_client import BaseClient
from .fedavg import FedAvgClient
from .fedprox import FedProxClient
from .fedlwf import FedLwFClient
from .scaffold import ScaffoldClient
from .peravg import PerAvgClient
from .pfedme import pFedMeClient
from .dcfcl import DCFCLClient
from .colledge import CollEdgeClient

__all__ = [
    'BaseClient',
    'FedAvgClient',
    'FedProxClient',
    'FedLwFClient',
    'ScaffoldClient',
    'PerAvgClient',
    'pFedMeClient',
    'DCFCLClient',
    'CollEdgeClient',
    'create_client',
]

# 算法名称到客户端类的映射
_ALGORITHM_MAP = {
    'FedAvg':    FedAvgClient,
    'Local':     FedAvgClient,    # Local 算法逻辑与 FedAvg 完全相同
    'FedProx':   FedProxClient,
    'FedLwF':    FedLwFClient,
    'SCAFFOLD':  ScaffoldClient,
    'PerAvg':    PerAvgClient,
    'pFedMe':    pFedMeClient,
    'ClusterFL': FedAvgClient,    # ClusterFL 客户端训练同 FedAvg，联盟形成在 Server 端
    'DCFCL':     DCFCLClient,
    'CollEdge': CollEdgeClient,
}


def create_client(config, client_id: int, model, train_data, test_data,
                  label_info: dict, unique_labels: int) -> BaseClient:
    """
    工厂函数：根据 config.algorithm 创建对应的联邦学习客户端。

    Args:
        config:       配置对象（包含 algorithm 字段）
        client_id:    客户端唯一 ID
        model:        全局模型（用于初始化本地模型）
        train_data:   训练数据集
        test_data:    测试数据集
        label_info:   标签信息字典（包含 'labels' 和 'counts'）
        unique_labels: 全局类别总数

    Returns:
        对应算法的 BaseClient 子类实例

    Raises:
        ValueError: 当 config.algorithm 不在支持列表中时
    """
    algorithm = getattr(config, 'algorithm', None)
    cls = _ALGORITHM_MAP.get(algorithm)
    if cls is None:
        supported = list(_ALGORITHM_MAP.keys())
        raise ValueError(
            f"不支持的算法: '{algorithm}'。"
            f"支持的算法: {supported}"
        )
    return cls(
        client_id=client_id,
        config=config,
        model=model,
        train_data=train_data,
        test_data=test_data,
        label_info=label_info,
        unique_labels=unique_labels,
    )
