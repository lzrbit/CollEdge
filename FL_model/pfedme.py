#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pFedMe 客户端实现。

通过 Moreau Envelope 正则化实现个性化联邦学习，
为每个客户端维护一个独立的个性化本地模型。

参考：Personalized Federated Learning using Hypernetworks
      (Shamsian et al., 2021) — pFedMe variant:
      Personalized Federated Learning with Moreau Envelopes
      (Dinh et al., 2020)
"""

import copy
from typing import Dict, Any
from .base_client import BaseClient


class pFedMeClient(BaseClient):
    """
    pFedMe 客户端。

    维护两个模型：
    - self.model:       全局模型（参与聚合）
    - self.local_model: 个性化本地模型（不参与聚合）

    训练时先对个性化模型做 K 步近端优化，再更新全局模型。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # pFedMe 专属：个性化本地模型
        self.local_model = copy.deepcopy(self.model)

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        pFedMe 本地训练。

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

            # K 步个性化近端优化
            for _ in range(self.config.K):
                _, _, logits = self.model(x)
                loss = self.ce_loss(logits, y.long())
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step(self.local_model.parameters())

            # 更新本地个性化模型（Moreau Envelope 近端步骤）
            for local_w, w in zip(self.local_model.parameters(), self.model.parameters()):
                local_w.data = (
                    local_w.data
                    - self.config.lamda * self.config.lr * (local_w.data - w.data)
                )

            total_loss += loss.item()

        # 将个性化模型参数上传（用于聚合时计算实际贡献）
        self.set_parameters(self.local_model)

        return {'loss': total_loss / self.config.local_epochs}
