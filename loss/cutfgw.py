from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _pairwise_cosine_distance(tokens: Tensor, mask: Tensor) -> Tensor:
    # 计算同一样本内部 token 两两之间的余弦距离，无效 padding 位置置零。
    normalized = F.normalize(tokens, p=2, dim=-1, eps=1.0e-8)
    similarity = torch.bmm(normalized, normalized.transpose(1, 2))
    distance = (1.0 - similarity).clamp(min=0.0, max=2.0)
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    return distance * pair_mask


def _temporal_differences(tokens: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor]:
    # 用相邻 token 差分描述动态变化，首个位置保留原始 token 作为起点。
    differences = torch.zeros_like(tokens)
    differences[:, 0] = tokens[:, 0]
    if tokens.size(1) > 1:
        differences[:, 1:] = tokens[:, 1:] - tokens[:, :-1]
    difference_mask = mask.clone()
    if mask.size(1) > 1:
        difference_mask[:, 1:] = mask[:, 1:] & mask[:, :-1]
    return differences, difference_mask


def _normalize_relation_matrix(matrix: Tensor, mask: Tensor) -> Tensor:
    # 按有效 pair 的平均尺度归一化，避免不同 batch 样本的距离量级差异过大。
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    denominator = pair_mask.sum(dim=(1, 2)).clamp_min(1)
    scale = (matrix * pair_mask).sum(dim=(1, 2)) / denominator
    return matrix / scale.detach().clamp_min(1.0e-4).view(-1, 1, 1)


def _linearized_gw_cost(teacher_relation: Tensor, student_relation: Tensor, plan: Tensor) -> Tensor:
    # 固定当前 transport plan 后，计算一次线性化的 GW 结构匹配代价。
    teacher_marginal = plan.sum(dim=-1)
    student_marginal = plan.sum(dim=-2)
    teacher_term = torch.bmm(
        teacher_relation.square(), teacher_marginal.unsqueeze(-1)
    ).squeeze(-1).unsqueeze(-1)
    student_term = torch.bmm(
        student_relation.square(), student_marginal.unsqueeze(-1)
    ).squeeze(-1).unsqueeze(1)
    cross_term = torch.bmm(
        torch.bmm(teacher_relation, plan), student_relation.transpose(1, 2)
    )
    return (teacher_term + student_term - 2.0 * cross_term).clamp_min(0.0)


class CausalUnbalancedTemporalFGW(nn.Module):
    """Causal unbalanced fused Gromov-Wasserstein relation distillation.

    The teacher support is a variable-length temporal memory trajectory. The
    student support consists of a small fixed number of ordered relation slots.
    Entropic unbalanced Sinkhorn is used inside each proximal FGW iteration.
    """

    def __init__(
        self,
        fused_alpha: float = 0.20,
        memory_weight: float = 1.0,
        dynamics_weight: float = 0.5,
        entropy: float = 0.08,
        marginal_relaxation: float = 0.8,
        time_prior_scale: float = 0.20,
        recency_decay: float = 1.0,
        learned_mass_blend: float = 0.5,
        fgw_iterations: int = 4,
        sinkhorn_iterations: int = 20,
    ) -> None:
        super().__init__()
        if not 0.0 <= fused_alpha <= 1.0:
            raise ValueError("fused_alpha must be in [0, 1].")
        if entropy <= 0.0 or marginal_relaxation <= 0.0 or time_prior_scale <= 0.0:
            raise ValueError("OT regularization parameters must be positive.")
        self.fused_alpha = fused_alpha
        self.memory_weight = memory_weight
        self.dynamics_weight = dynamics_weight
        self.entropy = entropy
        self.marginal_relaxation = marginal_relaxation
        self.time_prior_scale = time_prior_scale
        self.recency_decay = recency_decay
        self.learned_mass_blend = learned_mass_blend
        self.fgw_iterations = fgw_iterations
        self.sinkhorn_iterations = sinkhorn_iterations

    def _unbalanced_sinkhorn(
        self,
        cost: Tensor,
        teacher_mass: Tensor,
        student_mass: Tensor,
        prior: Tensor,
    ) -> Tensor:
        # 非平衡 Sinkhorn：允许边际质量轻微偏离，适配 teacher 变长历史和 student 固定槽。
        exponent = self.marginal_relaxation / (self.marginal_relaxation + self.entropy)
        support = prior > 0.0
        negative_large = torch.full((), -1.0e4, device=cost.device, dtype=cost.dtype)
        log_prior = torch.where(
            support,
            prior.clamp_min(1.0e-30).log(),
            negative_large,
        )
        log_kernel = log_prior - cost / self.entropy
        log_teacher_mass = torch.where(
            teacher_mass > 0.0,
            teacher_mass.clamp_min(1.0e-30).log(),
            negative_large,
        )
        log_student_mass = student_mass.clamp_min(1.0e-30).log()
        log_left = torch.zeros_like(teacher_mass)
        log_right = torch.zeros_like(student_mass)

        for _ in range(self.sinkhorn_iterations):
            # 在 log 空间交替更新左右缩放因子，提升数值稳定性。
            left_normalizer = torch.logsumexp(
                log_kernel + log_right.unsqueeze(-2), dim=-1
            )
            proposed_left = exponent * (log_teacher_mass - left_normalizer)
            log_left = torch.where(
                teacher_mass > 0.0,
                proposed_left,
                negative_large,
            )
            right_normalizer = torch.logsumexp(
                log_kernel + log_left.unsqueeze(-1), dim=-2
            )
            log_right = exponent * (log_student_mass - right_normalizer)

        log_plan = log_left.unsqueeze(-1) + log_kernel + log_right.unsqueeze(-2)
        plan = torch.exp(log_plan.clamp(min=-60.0, max=30.0))
        return plan * support

    def forward(
        self,
        teacher_tokens: Tensor,
        student_tokens: Tensor,
        teacher_ages: Tensor,
        student_ages: Tensor,
        teacher_mask: Tensor,
        teacher_mass: Optional[Tensor] = None,
        student_mass_logits: Optional[Tensor] = None,
        return_plan: bool = False,
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        teacher_mask = teacher_mask.bool()
        # 至少需要两个 teacher 历史 token 才能形成有意义的成对结构关系。
        valid_examples = teacher_mask.sum(dim=-1) >= 2
        if not torch.any(valid_examples):
            zero = student_tokens.sum() * 0.0
            if return_plan:
                return zero, {"transport": student_tokens.new_zeros(0)}
            return zero

        teacher_tokens = teacher_tokens[valid_examples]
        student_tokens = student_tokens[valid_examples]
        teacher_ages = teacher_ages[valid_examples]
        student_ages = student_ages[valid_examples]
        teacher_mask = teacher_mask[valid_examples]
        if teacher_mass is not None:
            teacher_mass = teacher_mass[valid_examples]
        if student_mass_logits is not None:
            student_mass_logits = student_mass_logits[valid_examples]

        # 将 teacher 的真实时间间隔压缩到 [0, 1] 附近，便于和 student 槽年龄比较。
        log_teacher_ages = torch.log1p(teacher_ages.clamp_min(0.0))
        maximum_age = (log_teacher_ages * teacher_mask).amax(dim=-1, keepdim=True).clamp_min(1.0e-6)
        normalized_teacher_ages = log_teacher_ages / maximum_age
        normalized_student_ages = student_ages.clamp(0.0, 1.0)
        attribute_cost = (
            normalized_teacher_ages.unsqueeze(-1) - normalized_student_ages.unsqueeze(-2)
        ).square()
        attribute_cost = attribute_cost * teacher_mask.unsqueeze(-1)

        if teacher_mass is None:
            # 没有 teacher 注意力质量时，用越新的历史越重要的先验作为质量分布。
            raw_teacher_mass = torch.exp(-self.recency_decay * normalized_teacher_ages)
            raw_teacher_mass = raw_teacher_mass * teacher_mask
        else:
            # 若 teacher 输出了关系质量，则优先使用；全空时回退到 recency 先验。
            raw_teacher_mass = teacher_mass.clamp_min(0.0) * teacher_mask
            fallback_mass = torch.exp(-self.recency_decay * normalized_teacher_ages) * teacher_mask
            empty_mass = raw_teacher_mass.sum(dim=-1, keepdim=True) <= 1.0e-8
            raw_teacher_mass = torch.where(empty_mass, fallback_mass, raw_teacher_mass)
        teacher_distribution = raw_teacher_mass / raw_teacher_mass.sum(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-8)

        slot_count = student_tokens.size(1)
        uniform_student_mass = torch.full_like(student_ages, 1.0 / float(slot_count))
        if student_mass_logits is None:
            student_distribution = uniform_student_mass
        else:
            # student 的槽质量由可学习 logits 和均匀分布混合，防止早期训练塌缩。
            learned_student_mass = torch.softmax(student_mass_logits, dim=-1)
            student_distribution = (
                self.learned_mass_blend * learned_student_mass
                + (1.0 - self.learned_mass_blend) * uniform_student_mass
            )

        teacher_memory_relation = _normalize_relation_matrix(
            _pairwise_cosine_distance(teacher_tokens, teacher_mask), teacher_mask
        )
        student_mask = torch.ones(
            student_tokens.shape[:2], device=student_tokens.device, dtype=torch.bool
        )
        student_memory_relation = _normalize_relation_matrix(
            _pairwise_cosine_distance(student_tokens, student_mask), student_mask
        )
        # memory_relation 对齐静态 token 几何，dynamic_relation 对齐相邻历史的变化模式。
        teacher_differences, teacher_difference_mask = _temporal_differences(
            teacher_tokens, teacher_mask
        )
        student_differences, student_difference_mask = _temporal_differences(
            student_tokens, student_mask
        )
        teacher_dynamic_relation = _normalize_relation_matrix(
            _pairwise_cosine_distance(teacher_differences, teacher_difference_mask),
            teacher_difference_mask,
        )
        student_dynamic_relation = _normalize_relation_matrix(
            _pairwise_cosine_distance(student_differences, student_difference_mask),
            student_difference_mask,
        )

        plan = teacher_distribution.unsqueeze(-1) * student_distribution.unsqueeze(-2)
        # 时间先验鼓励相近年龄的 teacher token 与 student 槽相互运输。
        prior = plan * torch.exp(-attribute_cost / self.time_prior_scale)
        prior = prior * teacher_mask.unsqueeze(-1)
        prior = prior / prior.sum(dim=(1, 2), keepdim=True).clamp_min(1.0e-30)

        for _ in range(self.fgw_iterations):
            # 交替计算结构代价并用 Sinkhorn 更新 transport plan，近似 FGW 优化。
            memory_cost = _linearized_gw_cost(
                teacher_memory_relation, student_memory_relation, plan
            )
            dynamics_cost = _linearized_gw_cost(
                teacher_dynamic_relation, student_dynamic_relation, plan
            )
            structural_cost = (
                self.memory_weight * memory_cost + self.dynamics_weight * dynamics_cost
            ) / max(self.memory_weight + self.dynamics_weight, 1.0e-8)
            cross_mask = teacher_mask.unsqueeze(-1).expand_as(structural_cost)
            structural_scale = (
                (structural_cost * cross_mask).sum(dim=(1, 2))
                / cross_mask.sum(dim=(1, 2)).clamp_min(1)
            ).detach().clamp_min(1.0e-4)
            structural_cost = structural_cost / structural_scale.view(-1, 1, 1)
            fused_cost = (
                self.fused_alpha * attribute_cost
                + (1.0 - self.fused_alpha) * structural_cost
            )
            plan = self._unbalanced_sinkhorn(
                fused_cost,
                teacher_distribution,
                student_distribution,
                prior,
            )

        # 使用最终 transport plan 重新计算一次代价，得到每个样本的关系蒸馏损失。
        memory_cost = _linearized_gw_cost(
            teacher_memory_relation, student_memory_relation, plan
        )
        dynamics_cost = _linearized_gw_cost(
            teacher_dynamic_relation, student_dynamic_relation, plan
        )
        structural_cost = (
            self.memory_weight * memory_cost + self.dynamics_weight * dynamics_cost
        ) / max(self.memory_weight + self.dynamics_weight, 1.0e-8)
        cross_mask = teacher_mask.unsqueeze(-1).expand_as(structural_cost)
        structural_scale = (
            (structural_cost * cross_mask).sum(dim=(1, 2))
            / cross_mask.sum(dim=(1, 2)).clamp_min(1)
        ).detach().clamp_min(1.0e-4)
        structural_cost = structural_cost / structural_scale.view(-1, 1, 1)
        final_cost = self.fused_alpha * attribute_cost + (1.0 - self.fused_alpha) * structural_cost
        transported_mass = plan.sum(dim=(1, 2)).clamp_min(1.0e-8)
        per_example_loss = (plan * final_cost).sum(dim=(1, 2)) / transported_mass
        loss = per_example_loss.mean()

        if return_plan:
            # diagnostics 只用于调试和可视化，不参与反向传播。
            diagnostics = {
                "transport": plan.detach(),
                "transport_mass": transported_mass.detach().mean(),
                "attribute_cost": (plan * attribute_cost).sum().detach()
                / plan.sum().detach().clamp_min(1.0e-8),
                "structural_cost": (plan * structural_cost).sum().detach()
                / plan.sum().detach().clamp_min(1.0e-8),
            }
            return loss, diagnostics
        return loss
