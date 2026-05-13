#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedAvg 客户端实现。

同时支持 Local 算法（逻辑相同，只是跳过服务端聚合步骤，
该区别由 Server 端控制，客户端训练逻辑一致）。

参考：Communication-Efficient Learning of Deep Networks
      from Decentralized Data (McMahan et al., 2017)
"""

from typing import Dict, Any
from .base_client import BaseClient


class FedAvgClient(BaseClient):
    """
    FedAvg / Local 客户端。

    每轮执行标准的本地 SGD 训练，然后将参数上传给服务端聚合。
    Local 模式下服务端不进行聚合，但客户端训练逻辑完全相同。
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        FedAvg 本地训练。

        执行 local_epochs 轮次的标准交叉熵训练。
        同时更新 differ（参数变化量），供 ClusterFL 等联盟算法使用。

        Args:
            glob_iter: 全局通信轮次
            task:      当前持续学习任务索引

        Returns:
            {'loss': 平均训练损失}
        """
        initial_params = self.param_vector.copy()
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()
            loss = self._train_step_basic(x, y)
            total_loss += loss

        # 更新 differ（供联盟相似度计算）
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params

        return {'loss': total_loss / self.config.local_epochs}
