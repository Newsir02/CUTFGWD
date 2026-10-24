"""Diagnose the TGN teacher's temporal influence kernel on a TGB dataset.

The diagnostic answers three questions before designing a distillation loss:
  1. Is the teacher's attention mass over historical edges informative
     (i.e. far from uniform / from a pure recency prior)?
  2. How much of that mass lands on the student's K most recent edges?
  3. Do teacher and student see exactly the same neighbor support, so that a
     support-matched distribution loss is well defined?
"""

from __future__ import annotations

import argparse
import json
from typing import Dict, List

import torch
from torch.nn import functional as F

from model import LightGNNStudent, PyGTGNTeacher
from train import (
    build_teacher,
    compute_data_signature,
    load_teacher_from_checkpoint,
    make_loaders,
    move_to_device,
    parse_args,
    resolve_device,
    resolve_student_seed,
    apply_method_configuration,
    validate_args,
)


def _teacher_kernel_stats(
    mass: torch.Tensor,
    ages: torch.Tensor,
    mask: torch.Tensor,
) -> Dict[str, float]:
    mask = mask.bool()
    rows = mask.any(dim=-1)
    if not bool(rows.any()):
        return {}
    mass = mass[rows]
    ages = ages[rows]
    mask = mask[rows]
    counts = mask.sum(dim=-1)
    n = counts.clamp_min(1)
    safe_mass = torch.where(mask, mass, torch.zeros_like(mass))
    safe_mass = safe_mass / safe_mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    log_mass = torch.where(
        mask, safe_mass.clamp_min(1.0e-12).log(), torch.zeros_like(safe_mass)
    )
    entropy = -(safe_mass * log_mass).sum(dim=-1) / n.float().log().clamp_min(1.0e-8)
    max_mass = torch.where(mask, safe_mass, torch.zeros_like(safe_mass)).amax(dim=-1)
    # Length-normalised TV distance from the uniform distribution. 1.0 = point mass.
    uniform = 1.0 / n.float()
    tv_uniform = 0.5 * torch.where(
        mask, (safe_mass - uniform.unsqueeze(-1)).abs(), torch.zeros_like(safe_mass)
    ).sum(dim=-1)

    # Recency prior: exp(-age) restricted to the valid support.
    prior = torch.where(mask, (-ages).exp(), torch.zeros_like(ages))
    prior = prior / prior.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    kl_uniform = (
        safe_mass * (safe_mass.clamp_min(1.0e-12).log() - uniform.unsqueeze(-1).log())
    )
    kl_uniform = torch.where(mask, kl_uniform, torch.zeros_like(kl_uniform)).sum(dim=-1)
    kl_recency = (
        safe_mass
        * (safe_mass.clamp_min(1.0e-12).log() - prior.clamp_min(1.0e-12).log())
    )
    kl_recency = torch.where(mask, kl_recency, torch.zeros_like(kl_recency)).sum(dim=-1)

    # Within-row Pearson correlation between recency (1 - normalised age) and mass.
    valid = mask.float()
    age_mean = (ages * valid).sum(-1, keepdim=True) / n.unsqueeze(-1)
    mass_mean = (safe_mass * valid).sum(-1, keepdim=True) / n.unsqueeze(-1)
    age_dev = (ages - age_mean) * valid
    mass_dev = (safe_mass - mass_mean) * valid
    covariance = (age_dev * mass_dev).sum(-1)
    age_var = (age_dev.square()).sum(-1)
    mass_var = (mass_dev.square()).sum(-1)
    correlation = covariance / (age_var * mass_var).sqrt().clamp_min(1.0e-8)

    # Mass share on the most recent 8 / oldest 8 valid edges.  Teacher relation
    # slots are left-aligned (0 = oldest valid, available-1 = newest), so the
    # newest slots are at a per-row offset rather than at the end of the tensor.
    available = mask.sum(dim=-1)
    newest8_share_list = []
    for k in (1, 4, 8):
        index = (
            available.unsqueeze(-1)
            - k
            + torch.arange(k, device=mass.device).unsqueeze(0)
        )
        valid_index = index >= 0
        index = index.clamp(0, mass.size(1) - 1)
        gathered = torch.gather(safe_mass, 1, index)
        newest8_share_list.append(
            float(
                (gathered * valid_index).sum(-1).mean()
            )
        )
    share_newest1, share_newest4, share_newest8 = newest8_share_list

    top_age = torch.where(
        mask, ages.masked_fill(~mask, float("inf")), torch.full_like(ages, float("inf"))
    ).amin(dim=-1)
    # Mean age of the highest-mass edge.
    top_index = safe_mass.argmax(dim=-1)
    top_mass_age = torch.gather(ages, 1, top_index.unsqueeze(-1)).squeeze(-1)

    return {
        "rows": float(mass.size(0)),
        "mean_valid_neighbors": float(n.float().mean()),
        "entropy_over_log_n": float(entropy.mean()),
        "max_mass": float(max_mass.mean()),
        "tv_from_uniform": float(tv_uniform.mean()),
        "kl_uniform": float(kl_uniform.mean()),
        "kl_recency": float(kl_recency.mean()),
        "mass_age_correlation": float(correlation.mean()),
        "share_on_newest1": share_newest1,
        "share_on_newest4": share_newest4,
        "share_on_newest8": share_newest8,
        "top_mass_age": float(top_mass_age.mean()),
        "min_age": float(top_age[torch.isfinite(top_age)].mean()),
    }


@torch.no_grad()
def diagnose(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    resolve_student_seed(args)
    apply_method_configuration(args)
    validate_args(args)
    bundle, train_loader, _, _ = make_loaders(args)
    signature = compute_data_signature(args, bundle)
    teacher = load_teacher_from_checkpoint(
        args.teacher_checkpoint, bundle, signature, device
    )
    teacher.eval()

    student = LightGNNStudent(
        node_features=(
            torch.cat(
                [bundle.node_features, bundle.structure_features], dim=-1
            )
            if args.structure_features
            else bundle.node_features
        ),
        hidden_dim=args.student_hidden,
        time_dim=args.time_dim,
        relation_dim=args.relation_dim,
        relation_slots=args.relation_slots,
        dropout=args.dropout,
        num_blocks=args.student_blocks,
        neighbors=args.student_neighbors,
        event_times=bundle.data.t,
        time_span=float(
            (bundle.data.t[-1].double() - bundle.data.t[0].double()).clamp_min(1.0)
        ),
    ).to(device)
    student.eval()

    time_span = (
        bundle.data.t[-1].double() - bundle.data.t[0].double()
    ).clamp_min(1.0)
    event_times = bundle.data.t.to(device).double()

    aggregate: List[Dict[str, float]] = []
    support_checks: List[float] = []
    eid_mismatch: List[float] = []
    support_counts: List[int] = []
    coverage: List[float] = []
    teacher.reset_state()
    student.reset_state()
    for step, raw_batch in enumerate(train_loader):
        if step >= args.batches:
            break
        batch = move_to_device(raw_batch, device)
        out = teacher.score_candidates(batch, return_relations=True)
        teacher.update_state(batch)
        stats = _teacher_kernel_stats(
            out["relation_mass"].float().cpu(),
            out["relation_ages"].float().cpu(),
            out["relation_mask"].cpu(),
        )
        if stats:
            aggregate.append(stats)

        # Student support for the same batch (before it observes the new edges).
        student(batch, return_relations=False)
        source = batch["src"].long()
        eid = student.neighbor_loader.e_id[source]
        k = int(student.neighbors)
        if eid.size(1) >= k:
            student_valid = eid[:, :k] >= 0
            teacher_available = out["relation_mask"].sum(dim=-1)
            rows = student_valid.all(dim=-1) & (teacher_available >= k)
            if bool(rows.any()):
                # LastNeighborLoader stores the newest edge first, while the
                # teacher exposes relation slots oldest -> newest.
                student_time = event_times[eid[rows, :k].clamp_min(0)].flip(-1)
                student_age = (
                    (
                        batch["event_time"].double()[rows].unsqueeze(-1)
                        - student_time
                    )
                    .clamp_min(0.0)
                    / time_span
                ).float()
                teacher_index = (
                    teacher_available[rows].unsqueeze(-1)
                    - k
                    + torch.arange(k, device=device).unsqueeze(0)
                )
                teacher_age = torch.gather(
                    out["relation_ages"].float()[rows], 1, teacher_index
                )
                teacher_eid = torch.gather(
                    out["relation_event_ids"][rows], 1, teacher_index
                )
                student_eid = eid[rows, :k].flip(-1)
                eid_mismatch.append(
                    float((teacher_eid != student_eid).float().mean())
                )
                diff = (student_age - teacher_age).abs()
                support_checks.append(float(diff.max()))
                support_counts.append(k)
                teacher_mass = torch.gather(
                    out["relation_mass"].float()[rows], 1, teacher_index
                )
                coverage.append(float(teacher_mass.sum(dim=-1).mean()))
        student.update_state(batch)
        if (step + 1) % 20 == 0:
            print(f"[diagnose] processed {step + 1} batches", flush=True)

    summary = {
        name: sum(item[name] for item in aggregate) / max(len(aggregate), 1)
        for name in aggregate[0]
    } if aggregate else {}
    summary["support_age_max_abs_diff"] = (
        max(support_checks) if support_checks else float("nan")
    )
    summary["support_match_fraction"] = (
        sum(1 for value in support_checks if value < 1.0e-5) / len(support_checks)
        if support_checks
        else float("nan")
    )
    summary["support_eid_mismatch"] = (
        sum(eid_mismatch) / len(eid_mismatch) if eid_mismatch else float("nan")
    )
    summary["teacher_mass_on_student_support"] = (
        sum(coverage) / len(coverage) if coverage else float("nan")
    )
    summary["student_support_k"] = support_counts[0] if support_counts else 0
    summary["batches"] = float(len(aggregate))
    print("KERNEL_DIAGNOSTIC " + json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--dataset", type=str, default="tgbl-wiki")
    parser.add_argument("--teacher-checkpoint", type=str, default="checkpoints/wiki-tgn/teacher.pt")
    parser.add_argument("--student-neighbors", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--structure-features", action="store_true")
    parser.add_argument("--structure-feature-set", type=str, default="basic")
    extra, _ = parser.parse_known_args()

    argv = [
        "--stage", "distill",
        "--dataset", extra.dataset,
        "--teacher-checkpoint", extra.teacher_checkpoint,
        "--student-arch", "gnn",
        "--student-neighbors", str(extra.student_neighbors),
        "--student-hidden", "32",
        "--student-blocks", "2",
        "--dropout", "0.3",
        "--student-seed", "42",
        "--device", extra.device,
        "--structure-feature-set", extra.structure_feature_set,
    ]
    if extra.structure_features:
        argv.append("--structure-features")
    args = parse_args(argv)
    args.batches = extra.batches
    diagnose(args)


if __name__ == "__main__":
    main()
