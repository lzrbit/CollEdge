#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SCAFFOLD 客户端实现。

使用控制变量（Control Variates）纠正客户端梯度偏差，
有效缓解非 IID 数据导致的客户端漂移问题。

参考：SCAFFOLD: Stochastic Controlled Averaging for Federated Learning
      (Karimireddy et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class ScaffoldClient(BaseClient):
    """
    SCAFFOLD 客户端。

    每轮训练结束后计算：
    - delta_model:   模型参数变化量（用于服务端聚合）
    - client_control: 更新后的本地控制变量
    - delta_control:  控制变量变化量（用于服务端聚合）
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        SCAFFOLD 本地训练。

        Args:
            glob_iter:      全局通信轮次
            task:           当前持续学习任务索引
            server_control: 服务端控制变量（必须提供）
            client_control: 所有客户端控制变量字典（必须提供）

        Returns:
            {'loss': 平均训练损失}
        """
        server_control = kwargs.get('server_control')
        client_control = kwargs.get('client_control')
        # 提取本客户端的控制变量
        my_control = client_control[self.id]

        global_model = copy.deepcopy(self.model)
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            _, _, logits = self.model(x)
            loss = self.ce_loss(logits, y.long())

            self.optimizer.zero_grad()
            loss.backward()
            # SCAFFOLD optimizer 会应用控制变量修正
            self.optimizer.step(server_control, my_control)

            total_loss += loss.item()

        # 计算模型差值和更新后的控制变量
        self.delta_model = self._compute_delta_model(global_model, self.model)
        self.client_control, self.delta_control = self._update_local_control(
            self.delta_model, server_control, my_control
        )

        return {'loss': total_loss / self.config.local_epochs}
