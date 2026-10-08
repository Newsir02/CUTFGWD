from __future__ import annotations

from typing import Dict, Tuple

from torch import Tensor, nn
from torch.nn import functional as F

from .cutfgw import CausalUnbalancedTemporalFGW
from .kd import (
    anchor_relation_loss,
    logit_kd_loss,
    rank_wasserstein_loss,
    relation_slot_diversity_loss,
)


class CUTFGWDDistillation(nn.Module):
    """Joint supervised, prediction, anchor-relation and CUT-FGW objective."""

    def __init__(
        self,
        task_weight: float = 1.0,
        logit_weight: float = 1.0,
        rank_weight: float = 0.5,
        relation_weight: float = 1.0,
        diversity_weight: float = 0.01,
        anchor_weight: float = 0.0,
        logit_temperature: float = 2.0,
        rank_temperature: float = 1.0,
        **cutfgw_kwargs: float,
    ) -> None:
        super().__init__()
        self.task_weight = task_weight
        self.logit_weight = logit_weight
        self.rank_weight = rank_weight
        self.relation_weight = relation_weight
        self.diversity_weight = diversity_weight
        self.anchor_weight = anchor_weight
        self.logit_temperature = logit_temperature
        self.rank_temperature = rank_temperature
        self.relation_loss = CausalUnbalancedTemporalFGW(**cutfgw_kwargs)

    def forward(
        self,
        student_output: Dict[str, Tensor],
        teacher_output: Dict[str, Tensor],
        target: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # task 是真实标签监督项，保证 student 仍直接优化下游排序目标。
        task = F.cross_entropy(student_output["logits"], target)
        # 以 student logits 为基准构造零标量：设备/dtype 一致，且权重为 0 时真正跳过计算。
        zero = student_output["logits"].sum() * 0.0

        # Prediction Distillation：对齐 teacher 与 student 的候选预测分布。
        if self.logit_weight != 0.0:
            logit = logit_kd_loss(
                student_output["logits"],
                teacher_output["logits"],
                self.logit_temperature,
            )
        else:
            logit = zero

        # TARD 的候选排序关系：对齐 teacher 的排序结构。
        if self.rank_weight != 0.0:
            rank = rank_wasserstein_loss(
                student_output["logits"],
                teacher_output["logits"],
                self.rank_temperature,
            )
        else:
            rank = zero

        # Anchor Relation Distillation：以年龄为锚点对齐关系表示。
        if self.anchor_weight != 0.0:
            anchor = anchor_relation_loss(
                student_output["relation_tokens"],
                student_output["relation_ages"],
                teacher_output["relation_tokens"],
                teacher_output["relation_ages"],
                teacher_output["relation_mask"],
            )
        else:
            anchor = zero

        # Structural OT Distillation：CUT-FGW 对齐变长关系轨迹和固定关系槽的结构。
        if self.relation_weight != 0.0:
            teacher_mass = teacher_output.get("relation_mass", None)
            if teacher_mass is not None:
                teacher_mass = teacher_mass.detach()
            relation = self.relation_loss(
                teacher_tokens=teacher_output["relation_tokens"].detach(),
                student_tokens=student_output["relation_tokens"],
                teacher_ages=teacher_output["relation_ages"].detach(),
                student_ages=student_output["relation_ages"],
                teacher_mask=teacher_output["relation_mask"],
                teacher_mass=teacher_mass,
                student_mass_logits=student_output.get(
                    "relation_mass_logits",
                    None,
                ),
            )
        else:
            relation = zero

        # 多样性正则避免所有关系槽学成同一个向量。
        if self.diversity_weight != 0.0:
            diversity = relation_slot_diversity_loss(
                student_output["relation_tokens"]
            )
        else:
            diversity = zero

        total = (
            self.task_weight * task
            + self.logit_weight * logit
            + self.rank_weight * rank
            + self.anchor_weight * anchor
            + self.relation_weight * relation
            + self.diversity_weight * diversity
        )
        # components 用于日志统计；detach 后避免日志路径保留计算图。
        components = {
            "total": total.detach(),
            "task": task.detach(),
            "logit": logit.detach(),
            "rank": rank.detach(),
            "anchor": anchor.detach(),
            "relation": relation.detach(),
            "diversity": diversity.detach(),
        }
        return total, components
