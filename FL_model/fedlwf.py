#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FedLwF 客户端实现。

结合联邦学习与 Learning without Forgetting（LwF）技术，
通过知识蒸馏减轻持续学习场景下的灾难性遗忘。

参考：Learning without Forgetting (Li & Hoiem, 2017)
      Applied to federated continual learning setting.
"""

import torch
from typing import Dict, Any
from .base_client import BaseClient


class FedLwFClient(BaseClient):
    """
    FedLwF 客户端。

    损失函数：L = L_CE + alpha * L_KD
    其中 L_KD 为当前模型与上一任务模型之间的蒸馏损失。
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        FedLwF 本地训练。

        当存在历史模型（if_last_copy=True）且不在第一个任务时，
        附加知识蒸馏损失防止遗忘。

        Args:
            glob_iter: 全局通信轮次
            task:      当前持续学习任务索引

        Returns:
            {'loss': 平均训练损失}
        """
        self.model.train()
        total_loss = 0.0

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            output, _, logits = self.model(x)
            class_loss = self.ce_loss(logits, y.long())

            if self.if_last_copy and self.current_task > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                # 蒸馏损失（与原始实现保持一致，温度指数 exp=0.5）
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)
                loss = class_loss + self.config.alpha * kd_loss
            else:
                loss = class_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return {'loss': total_loss / self.config.local_epochs}
