#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedProx 客户端实现。

在 FedAvg 基础上增加近端正则项（Proximal Term），约束本地模型
不偏离全局模型过远，提升非 IID 数据下的收敛稳定性。

参考：Federated Optimization in Heterogeneous Networks
      (Li et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class FedProxClient(BaseClient):
    """
    FedProx 客户端。

    损失函数：L = L_CE + (mu/2) * ||w - w_global||_2
    其中近端项使用 L2 范数（非平方），与原论文官方实现保持一致。
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        FedProx 本地训练。

        Args:
            glob_iter: 全局通信轮次
            task:      当前持续学习任务索引

        Returns:
            {'loss': 平均训练损失}
        """
        # 保存全局模型用于近端项计算
        global_model = copy.deepcopy(self.model)
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())

            # 近端正则项：L2 范数（与官方实现一致，非平方）
            proximal_term = 0.0
            for w, w_g in zip(self.model.parameters(), global_model.parameters()):
                proximal_term += (w - w_g).norm(2)
            loss += (self.config.mu / 2) * proximal_term

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / self.config.local_epochs}
