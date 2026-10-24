"""Does the TGN teacher's attention concentration predict link-prediction quality?

If the teacher's temporal influence distribution carries task-relevant
relational knowledge, rows with more concentrated attention should be easier
(higher MRR).  We also report the spread of mass over the newest 8 edges.
"""

from __future__ import annotations

import json
from typing import List, Tuple

import torch
from torch_geometric.loader import TemporalDataLoader

from train import (
    compute_data_signature,
    load_teacher_from_checkpoint,
    make_loaders,
    move_to_device,
    parse_args,
    replay_teacher,
    resolve_device,
)
from util.tgb_evaluation import _chunk_candidate_batch, _negative_tensors


@torch.no_grad()
def main() -> None:
    args = parse_args(
        [
            "--stage",
            "distill",
            "--dataset",
            "tgbl-wiki",
            "--teacher-checkpoint",
            "checkpoints/wiki-tgn/teacher.pt",
            "--student-arch",
            "gnn",
            "--student-seed",
            "42",
            "--device",
            "auto",
        ]
    )
    device = resolve_device(args.device)
    bundle, train_loader, val_loader, _ = make_loaders(args)
    teacher = load_teacher_from_checkpoint(
        args.teacher_checkpoint,
        bundle,
        compute_data_signature(args, bundle),
        device,
    )
    teacher.eval()
    teacher.reset_state()
    replay_teacher(teacher, [train_loader, val_loader], device)

    backend = bundle.tgb_dataset
    backend.load_test_ns()
    sampler = backend.negative_sampler
    metric = str(backend.eval_metric)
    from tgb.linkproppred.evaluate import Evaluator

    evaluator = Evaluator(name=bundle.dataset_name)
    loader = TemporalDataLoader(bundle.test_data, batch_size=args.batch_size)
    rows: List[Tuple[float, float, float, float, float]] = []
    for positive_batch in loader:
        query_kwargs = {}
        if hasattr(positive_batch, "edge_type"):
            query_kwargs["edge_type"] = positive_batch.edge_type
        negative_batches = sampler.query_batch(
            positive_batch.src,
            positive_batch.dst,
            positive_batch.t,
            split_mode="test",
            **query_kwargs,
        )
        negative_tensors = _negative_tensors(negative_batches, device)
        count = int(positive_batch.num_events)
        for start in range(0, count, 16):
            indices = list(range(start, min(start + 16, count)))
            chunk = _chunk_candidate_batch(
                positive_batch, negative_tensors, indices, device
            )
            output = teacher.score_candidates(chunk, return_relations=True)
            mass = output["relation_mass"].float().cpu()
            mask = output["relation_mask"].cpu()
            ages = output["relation_ages"].float().cpu()
            available = mask.sum(dim=-1)
            safe = torch.where(mask, mass, torch.zeros_like(mass))
            safe = safe / safe.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
            log_mass = torch.where(
                mask, safe.clamp_min(1.0e-12).log(), torch.zeros_like(safe)
            )
            entropy = -(safe * log_mass).sum(dim=-1) / available.float().log().clamp_min(
                1.0e-8
            )
            max_mass = safe.amax(dim=-1)
            # Mass on the newest 8 valid slots (left-aligned teacher support).
            newest = torch.zeros_like(safe)
            for k in range(1, 9):
                index = (available - k).clamp_min(0)
                newest.scatter_(1, index.unsqueeze(-1), 1.0)
            newest = torch.where(mask, newest, torch.zeros_like(newest))
            coverage = (safe * newest).sum(dim=-1) / newest.sum(dim=-1).clamp_min(1.0)
            newest_age = torch.gather(
                ages, 1, (available - 1).clamp_min(0).unsqueeze(-1)
            ).squeeze(-1)
            for row, event in enumerate(indices):
                width = int(negative_tensors[event].numel())
                scores = output["logits"][row, : 1 + width].float().cpu()
                result = evaluator.eval(
                    {
                        "y_pred_pos": scores[:1],
                        "y_pred_neg": scores[1:],
                        "eval_metric": [metric],
                    }
                )
                rows.append(
                    (
                        float(result[metric]),
                        float(entropy[row]),
                        float(max_mass[row]),
                        float(coverage[row]),
                        float(newest_age[row]),
                        float(available[row]),
                    )
                )
        teacher.update_state(
            {
                "event_index": positive_batch.event_index.to(device),
                "src": positive_batch.src.to(device),
                "positive_dst": positive_batch.dst.to(device),
                "event_time": positive_batch.t.to(device),
                "edge_features": positive_batch.msg.to(device),
            }
        )

    values = torch.tensor(rows)
    mrr, entropy, max_mass, coverage, newest_age, available = values.unbind(dim=-1)
    # Correlation between attention entropy and MRR within history-length bins,
    # to separate the "cold node => concentrated attention => low MRR" confound.
    degree_bins = []
    for low, high in ((0, 5), (5, 10), (10, 15), (15, 1000)):
        selected = (available >= low) & (available < high)
        if int(selected.sum()) < 10:
            continue
        degree_bins.append(
            {
                "available_range": [low, high],
                "events": int(selected.sum()),
                "mean_entropy": float(entropy[selected].mean()),
                "mean_mrr": float(mrr[selected].mean()),
                "corr_entropy_mrr": float(
                    torch.corrcoef(
                        torch.stack([entropy[selected], mrr[selected]])
                    )[0, 1]
                ),
            }
        )
    # Entropy quartiles inside a mature-history subset (available >= 10).
    mature = available >= 10
    mature_order = torch.argsort(entropy[mature])
    mature_entropy = entropy[mature]
    mature_mrr = mrr[mature]
    mature_chunk = max(len(mature_order) // 4, 1)
    mature_quartiles = []
    for q in range(4):
        selected = mature_order[
            q * mature_chunk
            : ((q + 1) * mature_chunk if q < 3 else len(mature_order))
        ]
        mature_quartiles.append(
            {
                "quartile": q,
                "mean_entropy": float(mature_entropy[selected].mean()),
                "mean_mrr": float(mature_mrr[selected].mean()),
            }
        )
    # Quartiles by attention entropy (all events).
    order = torch.argsort(entropy)
    quartiles = []
    chunk = max(len(order) // 4, 1)
    for q in range(4):
        selected = order[q * chunk : (q + 1) * chunk if q < 3 else len(order)]
        quartiles.append(
            {
                "entropy_low_to_high": q,
                "mean_entropy": float(entropy[selected].mean()),
                "mean_mrr": float(mrr[selected].mean()),
                "mean_max_mass": float(max_mass[selected].mean()),
            }
        )
    summary = {
        "events": int(len(rows)),
        "teacher_mrr": float(mrr.mean()),
        "corr_entropy_mrr": float(
            torch.corrcoef(torch.stack([entropy, mrr]))[0, 1]
        ),
        "corr_maxmass_mrr": float(
            torch.corrcoef(torch.stack([max_mass, mrr]))[0, 1]
        ),
        "corr_coverage_mrr": float(
            torch.corrcoef(torch.stack([coverage, mrr]))[0, 1]
        ),
        "quartile_entropy": quartiles,
        "degree_bins": degree_bins,
        "mature_entropy_quartiles": mature_quartiles,
    }
    print("ATTENTION_VALUE " + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
