#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Per-FedAvg 客户端实现。

基于 MAML（Model-Agnostic Meta-Learning）的个性化联邦学习方法，
训练一个好的全局初始化，使每个客户端能够只用少量步骤快速适应。

参考：Personalized Federated Learning with Theoretical Guarantees:
      A Model-Agnostic Meta-Learning Approach (Fallah et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class PerAvgClient(BaseClient):
    """
    Per-FedAvg（Per-FedAvg）客户端。

    每轮训练包含两步：
    1. 标准前向-反向更新（内层循环）
    2. 基于第二步梯度的个性化更新（外层循环，step size = beta）
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        Per-FedAvg 本地训练。

        Args:
            glob_iter: 全局通信轮次
            task:      当前持续学习任务索引

        Returns:
            {'loss': 平均训练损失}
        """
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            # 第一步：标准训练，临时保存模型状态
            temp_model = copy.deepcopy(self.model)
            x, y = self._get_next_batch()
            self._train_step_basic(x, y)

            # 第二步：个性化更新
            x, y = self._get_next_batch()
            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())

            self.optimizer.zero_grad()
            loss.backward()

            # 恢复到第一步前的模型参数，再应用个性化步骤
            for old_p, new_p in zip(self.model.parameters(), temp_model.parameters()):
                old_p.data = new_p.data.clone()
            self.optimizer.step(beta=self.config.beta)

            total_loss += loss.item()

        return {'loss': total_loss / self.config.local_epochs}
