#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CollEdge 客户端实现。

在 DCFCL 基础上引入 Dark Experience Replay++（DER++）机制，
通过维护回放缓冲区（Replay Buffer）存储历史样本的 logits 和标签，
在训练时同时对当前数据和历史回放数据进行优化。

完整损失：
  L = L_CE + lambda_kd * L_KD + lambda_proto_aug * L_proto
    + alpha_der * L_DER      （暗经验回放：MSE 匹配存储 logits）
    + beta_der  * L_ER       （经验回放：CE 使用真实标签）

参考：Dark Experience for General Continual Learning: a Strong,
      Simple Baseline (Buzzega et al., 2020)
"""

import copy
import torch
import torch.nn.functional as F
from typing import Dict, Any
from .base_client import BaseClient
from core.replay_buffer import ReplayBuffer


class CollEdgeClient(BaseClient):
    """
    CollEdge 客户端。

    在 DCFCL 三路损失基础上叠加 DER++ 回放损失，
    实现更强的抗遗忘能力。

    回放缓冲区跨轮次和任务持久保存，每轮训练后更新。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # CollEdge 专属：DER++ 回放缓冲区
        self.replay_buffer = ReplayBuffer(self.config.buffer_size, self.device)

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        CollEdge 本地训练（DCFCL + DER++）。

        Args:
            glob_iter:   全局通信轮次
            task:        当前持续学习任务索引
            proto_queue: 全局原型队列（用于原型增强损失）

        Returns:
            {
                'loss':             平均训练损失,
                'num_sample_class': 各类别样本计数字典
            }
        """
        proto_queue = kwargs.get('proto_queue')
        self.model.train()
        total_loss = 0.0
        num_sample_class = {k: 0 for k in range(self.config.num_classes)}

        der_alpha = self.config.der_alpha
        der_beta = self.config.der_beta

        # 保存教师模型（与 DCFCL 一致，每轮更新）
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True

        initial_params = self._get_param_vector().copy()

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            for label in y.tolist():
                num_sample_class[label] += 1

            # ---- 当前数据前向传播 ----
            output, features, logits = self.model(x)
            ce_loss = self.ce_loss(logits, y.long())

            # ---- 知识蒸馏损失（与 DCFCL 相同） ----
            kd_loss = 0.0
            if self.if_last_copy and self.current_task > 0 and self.config.lambda_kd > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)

            # ---- 原型增强损失（与 DCFCL 相同） ----
            proto_loss = 0.0
            if proto_queue is not None and self.config.lambda_proto_aug > 0:
                proto_loss = self._compute_proto_aug_loss(proto_queue)

            # ---- DER++ 回放损失 ----
            der_logit_loss = 0.0
            der_ce_loss = 0.0
            if self.config.use_der and not self.replay_buffer.is_empty():
                buf_x, buf_y, buf_stored_logits = self.replay_buffer.get_data(
                    self.config.batch_size
                )
                _, _, buf_logits = self.model(buf_x)

                # DER：当前 logits 匹配存储的历史 logits（暗经验）
                if der_alpha > 0:
                    der_logit_loss = F.mse_loss(buf_logits, buf_stored_logits)

                # DER++：对回放样本使用真实标签的 CE 损失
                if der_beta > 0:
                    der_ce_loss = self.ce_loss(buf_logits, buf_y)

            # ---- 联合损失 ----
            loss = (ce_loss
                    + self.config.lambda_kd * kd_loss
                    + self.config.lambda_proto_aug * proto_loss
                    + der_alpha * der_logit_loss
                    + der_beta * der_ce_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # ---- 将当前批次数据存入回放缓冲区 ----
            if self.config.use_der:
                with torch.no_grad():
                    _, _, store_logits = self.model(x)
                self.replay_buffer.add_data(x.detach(), y.detach(), store_logits.detach())

            total_loss += loss.item()

        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params

        return {
            'loss': total_loss / self.config.local_epochs,
            'num_sample_class': num_sample_class,
        }
