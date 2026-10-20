from __future__ import annotations

from typing import Dict

import torch
from torch import Tensor, nn


class HybridTeacher(nn.Module):
    """强教师的 logits + TGN 教师的时序关系 token。

    用于隔离 OT 的关系通道：logit/rank 项由打分更强的教师提供，
    关系 token 仍来自带真实历史邻居的 TGN 教师。
    """

    def __init__(
        self,
        logit_teacher: nn.Module,
        relation_teacher: nn.Module,
    ) -> None:
        super().__init__()
        self.logit_teacher = logit_teacher
        self.relation_teacher = relation_teacher
        self.hidden_dim = int(getattr(logit_teacher, "hidden_dim"))
        self.relation_dim = int(getattr(relation_teacher, "relation_dim"))
        teacher_layers = getattr(logit_teacher, "teacher_layers", None)
        if teacher_layers is not None:
            self.teacher_layers = int(teacher_layers)

    @torch.no_grad()
    def reset_state(self) -> None:
        for teacher in (self.logit_teacher, self.relation_teacher):
            reset = getattr(teacher, "reset_state", None)
            if callable(reset):
                reset()

    @torch.no_grad()
    def update_state(self, batch: Dict[str, Tensor]) -> None:
        for teacher in (self.logit_teacher, self.relation_teacher):
            update = getattr(teacher, "update_state", None)
            if callable(update):
                update(batch)

    def forward(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        output = {
            "logits": self.logit_teacher(
                batch,
                return_relations=False,
            )["logits"]
        }
        if not return_relations:
            return output
        scorer = getattr(self.relation_teacher, "score_candidates", None)
        if callable(scorer):
            relation = scorer(batch, return_relations=True)
        else:
            relation = self.relation_teacher(batch, return_relations=True)
        for key in (
            "relation_tokens",
            "relation_ages",
            "relation_mask",
            "relation_mass",
        ):
            if key in relation:
                output[key] = relation[key]
        if "source_embedding" in relation:
            output["source_embedding"] = relation["source_embedding"]
        return output

    def score_candidates(
        self,
        batch: Dict[str, Tensor],
        return_relations: bool = True,
    ) -> Dict[str, Tensor]:
        return self.forward(batch, return_relations=return_relations)
