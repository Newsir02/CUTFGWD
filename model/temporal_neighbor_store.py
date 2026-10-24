from __future__ import annotations

from typing import Tuple

import torch
from torch import Tensor
from torch_geometric.nn.models.tgn import LastNeighborLoader


class PyGNeighborStore:
    """Backward-compatible wrapper around PyG's ``LastNeighborLoader``.

    Kept for reproducing historical results.  It inherits the ring-buffer
    overwrite behaviour when a node has more than ``size`` events in a batch.
    """

    def __init__(self, num_nodes: int, size: int, device: torch.device | None = None) -> None:
        self.loader = LastNeighborLoader(num_nodes, size=size, device=device)

    @property
    def neighbors(self) -> Tensor:
        return self.loader.neighbors

    @property
    def e_id(self) -> Tensor:
        return self.loader.e_id

    @property
    def cur_e_id(self) -> int:
        return int(self.loader.cur_e_id)

    def reset_state(self) -> None:
        self.loader.reset_state()

    @torch.no_grad()
    def insert(self, src: Tensor, dst: Tensor, event_ids: Tensor | None = None) -> None:
        self.loader.insert(src.long(), dst.long())

    def query(self, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        node_ids = node_ids.long()
        return self.loader.neighbors[node_ids], self.loader.e_id[node_ids]



class ExactTemporalNeighborStore:
    """Keeps the *newest* K interactions per node with exact semantics.

    ``torch_geometric.nn.models.tgn.LastNeighborLoader`` writes new edges into
    dense ring-buffer positions before calling ``topk``.  When a node receives
    more than ``size`` events inside one batch, those writes overwrite each
    other and the retained set is not the newest set.  Hub nodes in TGB data
    (e.g. wiki) regularly trigger this, which makes the student's neighbour
    support differ from the teacher's.  This store keeps an explicit
    ``(event_id, neighbour)`` table and merges candidates by sorting on the
    event id, so the retained support is exactly the newest K events.
    """

    def __init__(self, num_nodes: int, size: int, device: torch.device | None = None) -> None:
        if num_nodes <= 0 or size <= 0:
            raise ValueError("num_nodes and size must be positive.")
        self.num_nodes = int(num_nodes)
        self.size = int(size)
        self.device = device
        self.neighbors = torch.full(
            (self.num_nodes, self.size), -1, dtype=torch.long, device=device
        )
        self.e_id = torch.full(
            (self.num_nodes, self.size), -1, dtype=torch.long, device=device
        )
        self.cur_e_id = 0

    def reset_state(self) -> None:
        self.neighbors.fill_(-1)
        self.e_id.fill_(-1)
        self.cur_e_id = 0

    @torch.no_grad()
    def insert(self, src: Tensor, dst: Tensor, event_ids: Tensor | None = None) -> None:
        src = src.long()
        dst = dst.long()
        batch_size = int(src.numel())
        if event_ids is None:
            event_ids = torch.arange(
                self.cur_e_id,
                self.cur_e_id + batch_size,
                device=src.device,
                dtype=torch.long,
            )
        else:
            event_ids = event_ids.long()
        nodes = torch.cat([src, dst])
        incoming = torch.cat([dst, src])
        incoming_eids = event_ids.repeat(2)
        self.cur_e_id += batch_size

        unique_nodes, inverse = torch.unique(nodes, return_inverse=True)
        # Node asc, event id desc (stable two-pass sort keeps hub ties deterministic).
        order = torch.argsort(-incoming_eids, stable=True)
        order = order[torch.argsort(inverse[order], stable=True)]
        inverse_sorted = inverse[order]
        eids_sorted = incoming_eids[order]
        neighbors_sorted = incoming[order]

        num_groups = int(unique_nodes.numel())
        counts = torch.zeros(num_groups, dtype=torch.long, device=nodes.device)
        counts.scatter_add_(0, inverse_sorted, torch.ones_like(inverse_sorted))
        starts = torch.cumsum(counts, dim=0) - counts
        rank = torch.arange(inverse_sorted.numel(), device=nodes.device) - starts[
            inverse_sorted
        ]
        keep = rank < self.size

        # At most size new events per node survive; place them in a padded table.
        new_eids = torch.full(
            (num_groups, self.size), -1, dtype=torch.long, device=nodes.device
        )
        new_neighbors = torch.full(
            (num_groups, self.size), -1, dtype=torch.long, device=nodes.device
        )
        target_group = inverse_sorted[keep]
        target_rank = rank[keep]
        new_eids[target_group, target_rank] = eids_sorted[keep]
        new_neighbors[target_group, target_rank] = neighbors_sorted[keep]

        old_eids = self.e_id[unique_nodes]
        old_neighbors = self.neighbors[unique_nodes]
        candidate_eids = torch.cat([old_eids, new_eids], dim=-1)
        candidate_neighbors = torch.cat([old_neighbors, new_neighbors], dim=-1)
        top_eids, permutation = candidate_eids.topk(self.size, dim=-1)
        top_neighbors = torch.gather(candidate_neighbors, 1, permutation)
        self.e_id[unique_nodes] = top_eids
        self.neighbors[unique_nodes] = top_neighbors

    def query(self, node_ids: Tensor) -> Tuple[Tensor, Tensor]:
        node_ids = node_ids.long()
        return self.neighbors[node_ids], self.e_id[node_ids]
