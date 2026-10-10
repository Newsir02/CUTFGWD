from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor
from torch.nn import functional as F


def logit_kd_loss(student_logits: Tensor, teacher_logits: Tensor, temperature: float = 2.0) -> Tensor:
    """Listwise KL distillation over candidates belonging to the same query."""

    # temperature 软化 teacher 分布，让 student 学到非目标候选之间的相对偏好。
    teacher_probability = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_log_probability = torch.log_softmax(student_logits / temperature, dim=-1)
    return F.kl_div(
        student_log_probability,
        teacher_probability,
        reduction="batchmean",
    ) * (temperature**2)


def rank_wasserstein_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
    top_weighted: bool = True,
) -> Tensor:
    """One-dimensional W1 distance on teacher-ordered candidate identities.

    Both distributions are gathered with the teacher permutation, so transport
    cannot obtain a small loss by matching two different candidate identities.
    """

    if student_logits.size(-1) < 2:
        return student_logits.sum() * 0.0
    # 以 teacher 的排序作为一维支撑，保证同一个候选身份之间做 transport。
    teacher_order = torch.argsort(teacher_logits.detach(), dim=-1, descending=True)
    teacher_probability = torch.softmax(teacher_logits.detach() / temperature, dim=-1)
    student_probability = torch.softmax(student_logits / temperature, dim=-1)
    teacher_ordered = torch.gather(teacher_probability, -1, teacher_order)
    student_ordered = torch.gather(student_probability, -1, teacher_order)

    # 对一维有序支撑上的分布，W1 等价于 CDF 差值的 L1 距离；最后一个 CDF 恒为 1。
    cdf_difference = torch.cumsum(teacher_ordered - student_ordered, dim=-1)[..., :-1].abs()
    if top_weighted:
        # 排名靠前的候选对推荐质量更重要，因此给前部 CDF 更高权重。
        positions = torch.arange(
            2,
            student_logits.size(-1) + 1,
            device=student_logits.device,
            dtype=student_logits.dtype,
        )
        weights = 1.0 / torch.log2(positions + 1.0)
        weights = weights / weights.mean().clamp_min(1.0e-8)
        cdf_difference = cdf_difference * weights
    return cdf_difference.mean()


def masked_relation_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    """Masked mean over the variable-length relation token axis."""

    weights = mask.unsqueeze(-1).to(tokens.dtype)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (tokens * weights).sum(dim=1) / denominator


def relation_alignment_loss(
    student_repr: Tensor,
    teacher_repr: Tensor,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Align anchor relation representations with cosine plus MSE."""

    target = teacher_repr.detach()
    if mask is not None:
        valid = mask.bool()
        if not bool(valid.any()):
            return student_repr.sum() * 0.0
        student_repr = student_repr[valid]
        target = target[valid]
    # cosine 对齐方向，MSE 对齐尺度，两者互补避免只学到方向或只学到幅度。
    cosine = 1.0 - F.cosine_similarity(student_repr, target, dim=-1, eps=1.0e-8)
    squared = F.mse_loss(student_repr, target, reduction="none").mean(dim=-1)
    return (cosine + squared).mean()


def anchor_relation_loss(
    student_tokens: Tensor,
    student_ages: Tensor,
    teacher_tokens: Tensor,
    teacher_ages: Tensor,
    teacher_mask: Tensor,
) -> Tensor:
    """Age-anchored relation representation distillation.

    Teacher relation tokens and student relation slots are both ordered by age.
    We use the normalized age as an anchor to build a stable correspondence:
    every student slot is matched to the valid teacher token with the closest
    age and then aligned in representation space. This transfers the teacher's
    relation content without requiring equal token counts.
    """

    teacher_mask = teacher_mask.bool()
    valid_rows = teacher_mask.any(dim=-1)
    if not bool(valid_rows.any()):
        return student_tokens.sum() * 0.0
    # 教师年龄按全局时间跨度归一化后可能非常小（近期历史都接近 0），学生槽年龄却
    # 遍布 [0,1]。必须逐样本把两侧年龄各自重标定到 [0,1]，否则所有学生槽会匹配到
    # 同一个（最近的）教师 token，锚点匹配退化为常数。
    positive_inf = torch.full(
        (),
        float("inf"),
        device=teacher_ages.device,
        dtype=teacher_ages.dtype,
    )
    negative_inf = torch.full(
        (),
        float("-inf"),
        device=teacher_ages.device,
        dtype=teacher_ages.dtype,
    )
    teacher_min = torch.where(
        teacher_mask, teacher_ages, positive_inf
    ).amin(dim=-1, keepdim=True)
    teacher_max = torch.where(
        teacher_mask, teacher_ages, negative_inf
    ).amax(dim=-1, keepdim=True)
    teacher_norm = (teacher_ages - teacher_min) / (
        teacher_max - teacher_min
    ).clamp_min(1.0e-6)
    teacher_norm = torch.where(
        teacher_mask,
        teacher_norm,
        torch.zeros_like(teacher_norm),
    )
    student_min = student_ages.amin(dim=-1, keepdim=True)
    student_max = student_ages.amax(dim=-1, keepdim=True)
    student_norm = (student_ages - student_min) / (
        student_max - student_min
    ).clamp_min(1.0e-6)
    # 年龄差的绝对值作为匹配代价，无效 teacher token 置为极大值后取最近邻。
    distance = (student_norm.unsqueeze(-1) - teacher_norm.unsqueeze(1)).abs()
    negative_large = torch.full(
        (),
        1.0e9,
        device=distance.device,
        dtype=distance.dtype,
    )
    distance = torch.where(teacher_mask.unsqueeze(1), distance, negative_large)
    nearest = distance.argmin(dim=-1)
    gather_index = nearest.unsqueeze(-1).expand(-1, -1, teacher_tokens.size(-1))
    target = torch.gather(teacher_tokens, 1, gather_index).detach()
    slot_valid = valid_rows.unsqueeze(1).expand_as(student_ages)
    cosine = 1.0 - F.cosine_similarity(student_tokens, target, dim=-1, eps=1.0e-8)
    squared = F.mse_loss(student_tokens, target, reduction="none").mean(dim=-1)
    return (cosine + squared)[slot_valid].mean()


def relation_slot_diversity_loss(slots: Tensor) -> Tensor:
    """Prevent train-only relation slots from collapsing to one vector."""

    if slots.size(1) < 2:
        return slots.sum() * 0.0
    # 惩罚不同槽之间的余弦相似度，促使槽学习互补的关系模式。
    normalized = F.normalize(slots, p=2, dim=-1, eps=1.0e-8)
    similarities = torch.bmm(normalized, normalized.transpose(1, 2))
    identity = torch.eye(slots.size(1), device=slots.device, dtype=slots.dtype).unsqueeze(0)
    off_diagonal = 1.0 - identity
    denominator = slots.size(0) * off_diagonal.sum().clamp_min(1.0)
    return ((similarities * off_diagonal) ** 2).sum() / denominator
