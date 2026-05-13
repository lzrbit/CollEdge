#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
BaseClient: 所有联邦学习客户端的基类。

包含数据加载、模型管理、原型计算、评估等所有算法共享的基础功能。
各算法子类只需实现 train() 方法即可。
"""

import copy
import math
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any

from core.optimizers import ScaffoldOptimizer, PerAvgOptimizer, pFedMeOptimizer

logger = logging.getLogger('DCFCL.Client')


class BaseClient:
    """
    联邦学习客户端基类。

    提供所有算法共享的基础功能：
    - 数据加载与管理
    - 模型初始化
    - 优化器配置
    - 原型计算（用于持续学习）
    - 模型参数同步
    - 评估方法

    子类需实现 train() 方法以定义具体的本地训练逻辑。
    """

    def __init__(self, client_id: int, config, model, train_data, test_data,
                 label_info: Dict, unique_labels: int):
        """
        初始化客户端。

        Args:
            client_id:    客户端唯一 ID
            config:       配置对象
            model:        全局模型（将深拷贝为本地模型）
            train_data:   训练数据集
            test_data:    测试数据集
            label_info:   标签信息字典（'labels' 和 'counts'）
            unique_labels: 全局类别总数
        """
        self.id = client_id
        self.config = config

        # 设备选择
        device_name = getattr(config, 'device', 'auto')
        if device_name == 'auto':
            device_name = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device_name == 'cuda' and not torch.cuda.is_available():
            device_name = 'cpu'
        self.device = torch.device(device_name)

        # 创建本地模型（深拷贝全局模型）
        # 延迟导入 core.models 以避免循环导入
        from core.models import create_model
        self.model = create_model(config)
        self.model.to(self.device)

        # 数据属性
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self.unique_labels = unique_labels

        # 创建 DataLoader
        self._setup_dataloaders()

        # 持续学习标签追踪
        self.classes_so_far: List[int] = list(label_info.get('labels', []))
        self.current_labels: List[int] = list(label_info.get('labels', []))
        self.classes_past_task: List[int] = []
        self.available_labels: List[int] = []
        self.available_labels_current: List[int] = []
        self.current_task: int = 0

        # 类别样本计数
        self.label_counts: Dict[int, int] = {i: 0 for i in range(unique_labels)}

        # 知识蒸馏（持续学习）
        self.last_copy: Optional[nn.Module] = None
        self.if_last_copy: bool = False

        # 设置优化器
        self._setup_optimizer()

        # 原型相关（DCFCL 系列）
        self.prototype = {"global": {}, "local": {}}
        self.radius = {"global": 0, "local": 0}
        self.feature_size = config.feature_size
        self.num_sample_class: Optional[Dict] = None

        # 梯度追踪（用于联盟相似度计算）
        self.param_vector = self._get_param_vector()
        self.differ = np.zeros_like(self.param_vector)
        self.l2_norm: Optional[float] = None

        # 测试数据追踪（持续学习跨任务评估）
        self.test_data_so_far = list(test_data)
        self.test_data_per_task = [test_data]

        # 损失函数
        ls = getattr(config, 'label_smoothing', 0.0)
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=ls)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    # =========================================================================
    # 初始化辅助方法
    # =========================================================================

    def _setup_dataloaders(self):
        """配置训练和测试 DataLoader。"""
        self.trainloader = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.testloader = DataLoader(
            self.test_data,
            batch_size=self.config.batch_size,
            drop_last=False
        )
        self.iter_trainloader = iter(self.trainloader)

    def _setup_optimizer(self):
        """根据算法配置优化器。"""
        if self.config.algorithm == 'SCAFFOLD':
            self.optimizer = ScaffoldOptimizer(
                self.model.parameters(),
                lr=self.config.scaffold_lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.algorithm == 'PerAvg':
            self.optimizer = PerAvgOptimizer(
                self.model.parameters(),
                lr=self.config.lr
            )
        elif self.config.algorithm == 'pFedMe':
            self.optimizer = pFedMeOptimizer(
                self.model.parameters(),
                lr=self.config.personal_lr,
                lamda=self.config.lamda
            )
        else:
            self.optimizer = self.model.classifier_optimizer

    # =========================================================================
    # 任务管理
    # =========================================================================

    def next_task(self, train_data, test_data, label_info: Dict):
        """
        为下一个持续学习任务更新客户端状态。

        Args:
            train_data: 新任务训练数据
            test_data:  新任务测试数据
            label_info: 新任务标签信息
        """
        # 保存当前模型用于知识蒸馏
        self.last_copy = copy.deepcopy(self.model)
        self.last_copy.to(self.device)
        self.if_last_copy = True

        # 更新数据
        self.train_data = train_data
        self.test_data = test_data
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)
        self._setup_dataloaders()

        # 更新标签追踪
        self.classes_past_task = list(self.classes_so_far)
        self.classes_so_far.extend(label_info.get('labels', []))
        self.current_labels = list(label_info.get('labels', []))

        # 更新测试数据
        self.test_data_so_far.extend(test_data)
        self.test_data_per_task.append(test_data)

        self.current_task += 1

    # =========================================================================
    # 训练接口（子类必须实现）
    # =========================================================================

    def train(self, glob_iter: int, task: int, **kwargs) -> Dict[str, Any]:
        """
        执行本地训练。

        Args:
            glob_iter: 全局迭代轮次
            task:      当前任务索引
            **kwargs:  算法特定参数

        Returns:
            包含训练结果的字典（至少包含 'loss' 键）
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} 必须实现 train() 方法。"
        )

    # =========================================================================
    # 共享训练工具
    # =========================================================================

    def _get_next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """获取下一个训练批次，自动重置迭代器。"""
        try:
            X, y = next(self.iter_trainloader)
        except StopIteration:
            self.iter_trainloader = iter(self.trainloader)
            X, y = next(self.iter_trainloader)
        return X.to(self.device), y.to(self.device)

    def _train_step_basic(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """基础训练步骤（前向 + 反向 + 梯度更新）。"""
        _, _, logits = self.model(x)
        loss = self.ce_loss(logits, y.long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def _cross_entropy_distill(self, outputs, targets, exp: float = 1.0, eps: float = 1e-5):
        """
        带温度缩放的知识蒸馏交叉熵损失。

        与原始代码实现保持一致。

        Args:
            outputs: 当前模型输出（softmax 概率）
            targets: 教师模型输出（softmax 概率）
            exp:     温度指数
            eps:     数值稳定性平滑项

        Returns:
            标量蒸馏损失
        """
        out = F.softmax(outputs, dim=1)
        tar = F.softmax(targets, dim=1)
        if exp != 1:
            out = out.pow(exp)
            out = out / out.sum(1).view(-1, 1).expand_as(out)
            tar = tar.pow(exp)
            tar = tar / tar.sum(1).view(-1, 1).expand_as(tar)
        out = out + eps / out.size(1)
        out = out / out.sum(1).view(-1, 1).expand_as(out)
        ce = -(tar * out.log()).sum(1)
        return ce.mean()

    def _compute_proto_aug_loss(self, proto_queue) -> torch.Tensor:
        """
        计算原型增强损失（Prototype Augmentation Loss）。

        通过在已有类别原型周围添加高斯噪声生成增强样本，
        防止模型遗忘旧任务的特征空间。

        Args:
            proto_queue: 全局原型队列对象

        Returns:
            标量原型增强损失
        """
        prototype = self.prototype.get("global", {}) or self.prototype.get("local", {})
        radius = self.radius.get("global", 0) or self.radius.get("local", 0)

        if not prototype or not radius:
            return torch.tensor(0.0).to(self.device)

        # 只对非当前任务的类别进行原型增强
        valid_indices = [k for k, v in prototype.items()
                         if np.sum(v) != 0 and k not in self.current_labels]

        if not valid_indices:
            return torch.tensor(0.0).to(self.device)

        proto_aug = []
        proto_aug_label = []
        for _ in range(self.config.batch_size):
            idx = np.random.choice(valid_indices)
            aug = prototype[idx] + np.random.normal(0, 1, self.feature_size) * radius
            proto_aug.append(aug)
            proto_aug_label.append(idx)

        proto_aug = torch.from_numpy(np.array(proto_aug)).float().to(self.device)
        proto_aug_label = torch.from_numpy(np.array(proto_aug_label)).long().to(self.device)

        output = self.model.fc(proto_aug)
        return self.ce_loss(output, proto_aug_label)

    # =========================================================================
    # 原型计算
    # =========================================================================

    def compute_prototypes(self) -> Tuple[float, Dict[int, np.ndarray], List[int]]:
        """
        从本地数据计算类别原型。

        Returns:
            Tuple of (radius, prototype_dict, class_labels)
        """
        self.model.eval()
        features_list = []
        labels_list = []

        with torch.no_grad():
            for _ in range(self.config.local_epochs):
                x, y = self._get_next_batch()
                feature = self.model.feature(x)
                features_list.append(feature.cpu().numpy())
                labels_list.append(y.cpu().numpy())

        features = np.concatenate(features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        feature_dim = features.shape[1]

        prototype = {}
        radius_dict = {}
        class_labels = []

        for cls in self.current_labels:
            idx = np.where(labels == cls)[0]
            class_labels.append(cls)

            if len(idx) == 0:
                prototype[cls] = np.zeros(self.feature_size)
            else:
                prototype[cls] = np.mean(features[idx], axis=0)

            if not self.prototype["local"]:
                if len(idx) > 1:
                    cov = np.cov(features[idx].T)
                    trace = np.trace(cov)
                    if not math.isnan(trace):
                        radius_dict[cls] = trace / feature_dim
                    else:
                        radius_dict[cls] = 0
                else:
                    radius_dict[cls] = 0

        if self.radius["local"]:
            radius = self.radius["local"]
        else:
            radius = np.sqrt(np.mean(list(radius_dict.values()))) if radius_dict else 0

        self.model.train()
        return radius, prototype, class_labels

    # =========================================================================
    # 参数同步与工具方法
    # =========================================================================

    def set_parameters(self, model, beta: float = 1.0):
        """
        从另一个模型同步参数，并刷新缓存状态。

        Args:
            model: 源模型
            beta:  EMA 混合系数（1.0 表示完全替换）
        """
        for old_param, new_param in zip(self.model.parameters(), model.parameters()):
            new_data = new_param.data.to(old_param.device)
            old_param.data = beta * new_data.clone() + (1 - beta) * old_param.data.clone()
        # 刷新缓存参数向量，避免联盟相似度使用过时权重
        self.param_vector = self._get_param_vector()
        self.differ = np.zeros_like(self.param_vector)

    def _get_param_vector(self) -> np.ndarray:
        """获取展平的参数向量。"""
        params = []
        for param in self.model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        return np.concatenate(params)

    def compute_l2_norm(self):
        """计算参数差异的 L2 范数。"""
        self.l2_norm = np.linalg.norm(self.differ, ord=2)

    def _compute_delta_model(self, model0, model1) -> Dict[str, torch.Tensor]:
        """计算两个模型之间的参数差值。"""
        delta = {}
        for name, param0 in model0.state_dict().items():
            param1 = model1.state_dict()[name]
            delta[name] = param0.detach() - param1.detach()
        return delta

    def _update_local_control(self, delta_model, server_control, my_control):
        """
        更新 SCAFFOLD 本地控制变量。

        Args:
            delta_model:    新旧模型参数差值
            server_control: 服务端控制变量（按参数名索引）
            my_control:     本客户端控制变量（按参数名索引）

        Returns:
            Tuple of (new_control, delta_control)
        """
        new_control = copy.deepcopy(my_control)
        delta_control = copy.deepcopy(my_control)

        for name in delta_model.keys():
            c = server_control[name].to(self.device)
            ci = my_control[name].to(self.device)
            delta = delta_model[name].to(self.device)

            new_ci = ci.data - c.data + delta.data / (
                self.config.local_epochs * self.config.scaffold_lr
            )
            new_control[name].data = new_ci
            delta_control[name].data = ci.data - new_ci

        return new_control, delta_control

    def get_sample_number(self) -> int:
        """返回训练样本数量。"""
        return self.train_samples

    def clean_up_counts(self):
        """重置标签计数。"""
        self.label_counts = {i: 0 for i in range(self.unique_labels)}

    # =========================================================================
    # 评估方法
    # =========================================================================

    def test(self) -> Tuple[float, float, int]:
        """
        在当前任务测试集上评估模型。

        Returns:
            Tuple of (correct_count, total_loss, total_samples)
        """
        self.model.eval()
        correct = 0
        total_loss = 0.0
        total = 0

        with torch.no_grad():
            for x, y in DataLoader(self.test_data, batch_size=self.config.batch_size):
                x, y = x.to(self.device), y.to(self.device)
                output, _, logits = self.model(x)
                total_loss += self.ce_loss(logits, y.long()).item()
                correct += (output.argmax(dim=1) == y).sum().item()
                total += y.size(0)

        return correct, total_loss, total

    def test_per_task(self) -> Tuple[List[int], List[float], List[int]]:
        """
        分任务评估模型准确率（评估持续学习遗忘程度）。

        Returns:
            Tuple of (correct_per_task, loss_per_task, samples_per_task)
        """
        self.model.eval()
        accs = []
        losses = []
        samples = []

        with torch.no_grad():
            for test_data in self.test_data_per_task:
                loader = DataLoader(test_data, batch_size=20)
                correct = 0
                loss = 0.0
                total = 0

                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output, _, logits = self.model(x)
                    loss += self.ce_loss(logits, y.long()).item()
                    correct += (output.argmax(dim=1) == y).sum().item()
                    total += y.size(0)

                accs.append(correct)
                losses.append(loss)
                samples.append(total)

        return accs, losses, samples
    
    def evaluate_emergence(self, all_test_data: List, all_labels_per_task: List[List[int]]) -> Dict:
        """
        评估涌现现象。
        
        涌现定义：客户端能够正确预测**本地从未见过的类别**。
        这种能力来自于其他客户端通过联邦协作传递的知识。
        
        Args:
            all_test_data: 所有任务的测试数据集列表（全局视角）
            all_labels_per_task: 每个任务的标签集列表（全局视角）
            
        Returns:
            字典包含:
            - 'unseen_correct': 对未见类别的正确预测数
            - 'unseen_total': 未见类别的总样本数
            - 'unseen_accuracy': 对未见类别的准确率（涌现率）
            - 'seen_correct': 对已见类别的正确预测数
            - 'seen_total': 已见类别的总样本数
            - 'seen_accuracy': 对已见类别的准确率
            - 'emergence_samples': 涌现样本列表 (x, y, pred, confidence)
            - 'per_class_emergence': 每个未见类别的统计 {class -> (correct, total)}
        """
        self.model.eval()
        
        # 获取本客户端本地见过的类别
        local_seen_classes = set(self.classes_so_far)
        
        # 统计指标
        unseen_correct = 0
        unseen_total = 0
        seen_correct = 0
        seen_total = 0
        
        emergence_samples = []  # 涌现样本
        per_class_emergence = {}  # 每个未见类别的准确率
        
        with torch.no_grad():
            for task_idx, test_data in enumerate(all_test_data):
                if len(test_data) == 0:
                    continue
                    
                loader = DataLoader(test_data, batch_size=20)
                
                for x, y in loader:
                    x, y = x.to(self.device), y.to(self.device)
                    output, features, logits = self.model(x)
                    probs = torch.softmax(logits, dim=1)
                    preds = output.argmax(dim=1)
                    confidences = probs.max(dim=1).values
                    
                    for i in range(len(y)):
                        true_label = y[i].item()
                        pred_label = preds[i].item()
                        confidence = confidences[i].item()
                        
                        if true_label in local_seen_classes:
                            # 本客户端本地见过的类别
                            seen_total += 1
                            if pred_label == true_label:
                                seen_correct += 1
                        else:
                            # 未见类别 - 涌现潜力！
                            unseen_total += 1
                            
                            if true_label not in per_class_emergence:
                                per_class_emergence[true_label] = {'correct': 0, 'total': 0}
                            per_class_emergence[true_label]['total'] += 1
                            
                            if pred_label == true_label:
                                unseen_correct += 1
                                per_class_emergence[true_label]['correct'] += 1
                                
                                # 记录涌现样本
                                emergence_samples.append({
                                    'sample': x[i].cpu().numpy(),
                                    'true_label': true_label,
                                    'pred_label': pred_label,
                                    'confidence': confidence,
                                    'task_idx': task_idx,
                                    'client_id': self.id,
                                    'local_seen_classes': list(local_seen_classes),
                                })
        
        return {
            'unseen_correct': unseen_correct,
            'unseen_total': unseen_total,
            'unseen_accuracy': unseen_correct / unseen_total if unseen_total > 0 else 0.0,
            'seen_correct': seen_correct,
            'seen_total': seen_total,
            'seen_accuracy': seen_correct / seen_total if seen_total > 0 else 0.0,
            'emergence_samples': emergence_samples,
            'per_class_emergence': per_class_emergence,
            'local_seen_classes': list(local_seen_classes),
        }
