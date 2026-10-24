from __future__ import annotations

import json

import torch

from train import (
    compute_data_signature,
    evaluate_teacher_from_history,
    load_teacher_from_checkpoint,
    make_loaders,
    parse_args,
    resolve_device,
)


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
    bundle, train_loader, val_loader, test_loader = make_loaders(args)
    signature = compute_data_signature(args, bundle)
    results = {}
    for neighbors in (8, 12, 16, 24):
        teacher = load_teacher_from_checkpoint(
            args.teacher_checkpoint,
            bundle,
            signature,
            device,
            temporal_neighbors=neighbors,
        )
        metrics = evaluate_teacher_from_history(
            teacher,
            bundle,
            [train_loader, val_loader],
            test_loader,
            "test",
            args,
            device,
        )
        results[str(neighbors)] = float(metrics[metrics["metric"]])
        print(f"teacher_neighbors={neighbors} test_mrr={results[str(neighbors)]:.4f}")
    print("TEACHER_SUPPORT_SWEEP " + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
