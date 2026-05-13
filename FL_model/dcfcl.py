#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DCFCL 客户端实现。

Decentralized Continual Federated Learning（DCFCL）论文提出的核心方法。
结合以下三种损失进行本地训练：
  - L_CE:         分类交叉熵损失
  - L_KD:         知识蒸馏损失（防止遗忘旧任务特征空间）
  - L_proto_aug:  原型增强损失（从旧类别原型生成虚拟样本）

完整损失（论文公式 6）：
  L = L_CE + lambda_kd * L_KD + lambda_proto_aug * L_proto_aug

训练结束后计算模型参数变化量（differ），用于联盟相似度评分。
"""

import copy
import torch
from typing import Dict, Any
from .base_client import BaseClient


class DCFCLClient(BaseClient):
    """
    DCFCL 客户端。

    实现论文 Section 4.1 中描述的本地训练流程：
    1. 每轮保存教师模型（用于轮间 KD，与官方代码保持一致）
    2. 三路损失联合优化
    3. 记录参数变化量供联盟形成使用
    """

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        DCFCL 本地训练。

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

        # 每轮保存教师模型用于 KD
        # 论文公式 4：教师为上一轮模型（而非仅任务切换时）
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True

        # 记录初始参数：differ = 所有步骤的总参数变化量
        initial_params = self._get_param_vector().copy()

        for _ in range(self.config.local_epochs):
            x, y = self._get_next_batch()

            # 追踪各类别样本数（用于原型加权聚合）
            for label in y.tolist():
                num_sample_class[label] += 1

            output, features, logits = self.model(x)

            # 分类损失（使用 logits，非 softmax 输出）
            ce_loss = self.ce_loss(logits, y.long())

            # 知识蒸馏损失（论文 Section 4.1，公式 4）
            kd_loss = 0.0
            if self.if_last_copy and self.current_task > 0 and self.config.lambda_kd > 0:
                with torch.no_grad():
                    old_output, _, _ = self.last_copy(x)
                kd_loss = self._cross_entropy_distill(output, old_output, exp=0.5)

            # 原型增强损失
            proto_loss = 0.0
            if proto_queue is not None and self.config.lambda_proto_aug > 0:
                proto_loss = self._compute_proto_aug_loss(proto_queue)

            # 联合损失（论文公式 6）
            loss = (ce_loss
                    + self.config.lambda_kd * kd_loss
                    + self.config.lambda_proto_aug * proto_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        # 计算总参数变化量（供联盟相似度评分使用）
        self.param_vector = self._get_param_vector()
        self.differ = self.param_vector - initial_params

        return {
            'loss': total_loss / self.config.local_epochs,
            'num_sample_class': num_sample_class,
        }
