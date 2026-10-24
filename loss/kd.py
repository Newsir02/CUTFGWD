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


def candidate_relation_structure_loss(
    student_embeddings: Tensor,
    teacher_embeddings: Tensor,
    temperature: float = 1.0,
    divergence: str = "kl",
) -> Tensor:
    """Distill time-conditioned relative similarity among candidates.

    Candidate embeddings produced by a temporal GNN encode each candidate's
    relation to the evolving stream.  Their pairwise cosine geometry is
    invariant to rotations/scaling and to the teacher/student hidden
    dimensions, so it can be compared directly.  Matching the row-wise
    similarity distribution transfers *relative* relational structure, which
    scalar logits only partially constrain.
    """

    if divergence not in {"kl", "mse"}:
        raise ValueError("candidate structure divergence must be kl or mse.")
    if student_embeddings.size(1) < 2:
        return student_embeddings.sum() * 0.0
    if temperature <= 0.0:
        raise ValueError("candidate structure temperature must be positive.")
    student = F.normalize(student_embeddings, p=2, dim=-1, eps=1.0e-8)
    teacher = F.normalize(teacher_embeddings.detach(), p=2, dim=-1, eps=1.0e-8)
    student_similarity = torch.bmm(student, student.transpose(1, 2))
    teacher_similarity = torch.bmm(teacher, teacher.transpose(1, 2))
    count = student_embeddings.size(1)
    off_diagonal = 1.0 - torch.eye(
        count, device=student_similarity.device, dtype=student_similarity.dtype
    ).unsqueeze(0)
    if divergence == "mse":
        difference = (student_similarity - teacher_similarity) * off_diagonal
        return difference.square().sum(dim=(1, 2)).div(
            (off_diagonal.sum(dim=(1, 2))).clamp_min(1.0)
        ).mean()
    teacher_logits = (teacher_similarity / temperature).masked_fill(
        off_diagonal == 0.0, -1.0e9
    )
    student_logits = (student_similarity / temperature).masked_fill(
        off_diagonal == 0.0, -1.0e9
    )
    teacher_probability = torch.softmax(teacher_logits, dim=-1)
    student_log_probability = torch.log_softmax(student_logits, dim=-1)
    kl = (
        teacher_probability
        * (
            teacher_probability.clamp_min(1.0e-12).log()
            - student_log_probability
        )
    )
    return kl.sum(dim=-1).mean()


def temporal_relational_drift_loss(
    student_source: Tensor,
    teacher_source: Tensor,
    source_ids: Tensor,
    event_index: Tensor,
    cache: dict,
    num_nodes: int,
    max_gap: int = 512,
    eps: float = 1.0e-6,
    randomize_teacher: bool = False,
) -> Tensor:
    """Match the *direction of change* of the source's relation representation.

    For a source that recurs across batches, the teacher's source embedding
    moves as its memory/neighbourhood evolves.  Its first difference is the
    relational drift.  Aligning the student's drift direction transfers the
    dynamics of the relation representation, not just a single snapshot.

    ``cache`` is a mutable dict owned by the training loop; it stores the last
    teacher/student source embedding, the last event index and validity per
    node.  It must be reset at the start of every epoch.
    """

    if student_source.shape != teacher_source.shape:
        raise ValueError(
            "drift distillation requires equal student/teacher source dims: "
            f"{student_source.shape[-1]} vs {teacher_source.shape[-1]}."
        )
    teacher_source = teacher_source.detach()
    source_ids = source_ids.long()
    event_index = event_index.long()
    if "teacher" not in cache:
        dim = int(student_source.size(-1))
        cache["teacher"] = student_source.new_zeros(num_nodes, dim)
        cache["student"] = student_source.new_zeros(num_nodes, dim)
        cache["event"] = torch.full(
            (num_nodes,), -1, dtype=torch.long, device=source_ids.device
        )
        cache["valid"] = torch.zeros(
            num_nodes, dtype=torch.bool, device=source_ids.device
        )

    previous_teacher = cache["teacher"][source_ids]
    previous_student = cache["student"][source_ids]
    previous_event = cache["event"][source_ids]
    gap = event_index - previous_event
    valid = (
        cache["valid"][source_ids]
        & (gap > 0)
        & (gap <= int(max_gap))
    )
    teacher_drift = teacher_source - previous_teacher
    if randomize_teacher and teacher_drift.size(0) > 1:
        # Control: keep the same drift magnitudes but break the source-wise
        # correspondence between the teacher direction and the student row.
        teacher_drift = teacher_drift[torch.randperm(teacher_drift.size(0), device=teacher_drift.device)]
    student_drift = student_source - previous_student
    teacher_norm = teacher_drift.norm(dim=-1)
    student_norm = student_drift.norm(dim=-1)
    valid = valid & (teacher_norm > eps) & (student_norm > eps)
    if bool(valid.any()):
        cosine = F.cosine_similarity(
            student_drift[valid], teacher_drift[valid], dim=-1, eps=1.0e-8
        )
        loss = (1.0 - cosine).mean()
    else:
        loss = student_source.sum() * 0.0

    # Update the cache with the last occurrence of each source in this batch.
    unique, inverse = torch.unique(source_ids, return_inverse=True)
    last_position = torch.full(
        (unique.numel(),), -1, dtype=torch.long, device=source_ids.device
    )
    positions = torch.arange(source_ids.numel(), device=source_ids.device)
    last_position.scatter_reduce_(0, inverse, positions, reduce="amax")
    cache["teacher"][unique] = teacher_source.detach()[last_position]
    cache["student"][unique] = student_source.detach()[last_position]
    cache["event"][unique] = event_index[last_position]
    cache["valid"][unique] = True
    return loss


def hard_negative_margin_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    target: Tensor,
    loss_type: str = "smooth_l1",
    pair_mode: str = "teacher",
) -> Tensor:
    """Distill the positive-vs-hardest-negative margin.

    The official MRR is decided by the positive's relation to the highest
    scoring negative.  Matching the teacher's *margin on that binding pair*
    is a local, task-aligned relational target; easy negatives contribute
    little and are downweighted implicitly.

    ``pair_mode='teacher'`` fixes the negative to the teacher's hardest one,
    so both margins refer to the same candidate pair; ``'student'`` uses each
    model's own hardest negative (legacy behaviour).
    """

    if loss_type not in {"smooth_l1", "l2"}:
        raise ValueError("margin loss_type must be smooth_l1 or l2.")
    if pair_mode not in {"teacher", "student"}:
        raise ValueError("margin pair_mode must be teacher or student.")
    if student_logits.size(-1) < 2:
        return student_logits.sum() * 0.0
    target = target.long()
    if bool((target < 0).any()) or bool((target >= student_logits.size(-1)).any()):
        raise ValueError("margin target is out of range.")
    position = F.one_hot(target, num_classes=student_logits.size(-1)).bool()
    negative_mask = ~position
    student_positive = torch.gather(
        student_logits, 1, target.unsqueeze(-1)
    ).squeeze(-1)
    teacher_positive = torch.gather(
        teacher_logits.detach(), 1, target.unsqueeze(-1)
    ).squeeze(-1)
    if pair_mode == "teacher":
        teacher_masked = teacher_logits.detach().masked_fill(
            ~negative_mask, -1.0e9
        )
        teacher_hard_index = teacher_masked.argmax(dim=-1)
        student_hard = torch.gather(
            student_logits, 1, teacher_hard_index.unsqueeze(-1)
        ).squeeze(-1)
        teacher_hard = torch.gather(
            teacher_masked, 1, teacher_hard_index.unsqueeze(-1)
        ).squeeze(-1)
    else:
        student_hard = student_logits.masked_fill(~negative_mask, -1.0e9).amax(
            dim=-1
        )
        teacher_hard = teacher_logits.detach().masked_fill(
            ~negative_mask, -1.0e9
        ).amax(dim=-1)
    student_margin = student_positive - student_hard
    teacher_margin = teacher_positive - teacher_hard
    if loss_type == "l2":
        return (student_margin - teacher_margin).square().mean()
    return F.smooth_l1_loss(student_margin, teacher_margin)


def support_exact_relation_content_loss(
    student_tokens: Tensor,
    teacher_tokens: Tensor,
    teacher_mask: Tensor,
) -> Tensor:
    """Align relation-token *content* on the shared newest-K event support.

    Teacher relation slots are ordered oldest -> newest and may contain more
    events than the student.  Because both observe the same stream, the newest
    ``R`` teacher tokens correspond exactly (by event order) to the ``R``
    student slots; no OT or age matching is needed.  Rows with fewer than R
    valid teacher tokens are skipped.
    """

    teacher_mask = teacher_mask.bool()
    if student_tokens.size(1) == 0 or teacher_tokens.size(1) == 0:
        return student_tokens.sum() * 0.0
    slots = int(student_tokens.size(1))
    if teacher_tokens.size(1) < slots:
        return student_tokens.sum() * 0.0
    available = teacher_mask.sum(dim=-1)
    valid_rows = available >= slots
    if not bool(valid_rows.any()):
        return student_tokens.sum() * 0.0
    index = (
        available[valid_rows].unsqueeze(-1)
        - slots
        + torch.arange(slots, device=teacher_tokens.device).unsqueeze(0)
    )
    target = torch.gather(
        teacher_tokens[valid_rows],
        1,
        index.unsqueeze(-1).expand(-1, -1, teacher_tokens.size(-1)),
    ).detach()
    student = student_tokens[valid_rows]
    if student.size(-1) != target.size(-1):
        raise ValueError(
            "relation content distillation requires equal relation dims: "
            f"{student.size(-1)} vs {target.size(-1)}."
        )
    cosine = 1.0 - F.cosine_similarity(student, target, dim=-1, eps=1.0e-8)
    # Cosine only: the absolute token scale differs across teacher/student
    # parameterizations; an unweighted MSE term would dominate the CE loss.
    return cosine.mean()


def pairwise_relation_kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    temperature: float = 1.0,
) -> Tensor:
    """Pairwise relational KD over candidates (soft RankNet).

    Instead of matching the full softmax, this matches the teacher's *pairwise
    preferences* between candidates with a logistic loss.  The target is a
    soft preference ``sigmoid((z_T_i - z_T_j) / temperature)``, which is
    invariant to per-event shifts and less sensitive to calibration than KL.
    """

    if student_logits.size(-1) < 2:
        return student_logits.sum() * 0.0
    if temperature <= 0.0:
        raise ValueError("pairwise temperature must be positive.")
    teacher_logits = teacher_logits.detach()
    student_difference = (
        student_logits.unsqueeze(-1) - student_logits.unsqueeze(-2)
    ) / temperature
    teacher_difference = (
        teacher_logits.unsqueeze(-1) - teacher_logits.unsqueeze(-2)
    ) / temperature
    target = torch.sigmoid(teacher_difference)
    count = student_logits.size(-1)
    off_diagonal = 1.0 - torch.eye(
        count, device=student_logits.device, dtype=student_logits.dtype
    ).unsqueeze(0)
    loss = F.binary_cross_entropy_with_logits(
        student_difference,
        target,
        reduction="none",
    )
    return (loss * off_diagonal).sum(dim=(1, 2)).div(
        off_diagonal.sum(dim=(1, 2)).clamp_min(1.0)
    ).mean()


def temporal_kernel_distillation_loss(
    student_mass: Tensor,
    student_event_ids: Tensor,
    student_mask: Tensor,
    teacher_mass: Tensor,
    teacher_event_ids: Tensor,
    teacher_mask: Tensor,
    divergence: str = "kl",
) -> Tensor:
    """Match the teacher's temporal influence distribution on shared support.

    Teacher and student observe the same event stream, so the historical
    interaction support is *known*: instead of estimating a transport plan
    between two unrelated token sets, we look up the teacher's mass for the
    exact event ids used by the student and compare the two conditional
    distributions.  This turns relation distillation into a direct
    distribution-matching problem over the teacher's temporal influence
    operator (a discrete conditional intensity / Hawkes-style kernel).

    Args:
        student_mass: ``[B, K]`` normalised student influence weights.
        student_event_ids: ``[B, K]`` event ids of the student support (-1 pad).
        student_mask: ``[B, K]`` valid student slots.
        teacher_mass: ``[B, N]`` normalised teacher attention mass.
        teacher_event_ids: ``[B, N]`` event ids of the teacher support.
        teacher_mask: ``[B, N]`` valid teacher slots.
        divergence: ``kl``, ``js`` or ``mse`` (attention-transfer style control).
    """

    if divergence not in {"kl", "js", "mse"}:
        raise ValueError("divergence must be one of: kl, js, mse.")
    student_mask = student_mask.bool()
    teacher_mask = teacher_mask.bool()
    if not bool(student_mask.any()) or not bool(teacher_mask.any()):
        return student_mass.sum() * 0.0
    if student_event_ids.shape != student_mass.shape:
        raise ValueError("student_event_ids must match student_mass shape.")
    if teacher_event_ids.shape != teacher_mass.shape:
        raise ValueError("teacher_event_ids must match teacher_mass shape.")

    # [B, K, N] exact-support match. Teacher ids are unique within a row, so
    # summing matched masses is equivalent to a lookup.
    matched = (
        student_event_ids.unsqueeze(-1) == teacher_event_ids.unsqueeze(-2)
    )
    matched = (
        matched
        & teacher_mask.unsqueeze(-2)
        & student_mask.unsqueeze(-1)
    )
    teacher_target = (
        teacher_mass.unsqueeze(-2)
        * matched.to(teacher_mass.dtype)
    ).sum(dim=-1)
    student_target = student_mass * matched.any(dim=-1).to(student_mass.dtype)

    teacher_total = teacher_target.sum(dim=-1, keepdim=True)
    student_total = student_target.sum(dim=-1, keepdim=True)
    valid = (teacher_total.squeeze(-1) > 1.0e-8) & (
        student_total.squeeze(-1) > 1.0e-8
    )
    if not bool(valid.any()):
        return student_mass.sum() * 0.0
    teacher_probability = teacher_target[valid] / teacher_total[valid]
    student_probability = student_target[valid] / student_total[valid]
    log_teacher = teacher_probability.clamp_min(1.0e-12).log()
    log_student = student_probability.clamp_min(1.0e-12).log()
    if divergence == "kl":
        loss = (teacher_probability * (log_teacher - log_student)).sum(dim=-1)
    elif divergence == "js":
        mixture = 0.5 * (teacher_probability + student_probability)
        log_mixture = mixture.clamp_min(1.0e-12).log()
        loss = 0.5 * (
            (teacher_probability * (log_teacher - log_mixture)).sum(dim=-1)
            + (student_probability * (log_student - log_mixture)).sum(dim=-1)
        )
    else:
        loss = (teacher_probability - student_probability).square().sum(dim=-1)
    return loss.mean()

