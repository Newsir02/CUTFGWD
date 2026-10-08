from __future__ import annotations

from pathlib import Path

import numpy as np

from tgb.utils.pre_process import load_edgelist_wiki


def test_py_tgb_wiki_loader_can_offset_destination_ids(tmp_path: Path) -> None:
    edge_file = tmp_path / "tiny_wiki.csv"
    edge_file.write_text(
        "user,item,time,label,feature\n"
        "0,0,1,0,0.1\n"
        "1,1,2,0,0.2\n",
        encoding="utf-8",
    )

    frame, messages, node_ids = load_edgelist_wiki(str(edge_file))

    assert np.array_equal(frame["u"].to_numpy(), np.array([0, 1]))
    assert np.array_equal(frame["i"].to_numpy(), np.array([2, 3]))
    assert messages.shape == (2, 1)
    assert node_ids is None


def test_tgb_root_argument_resolves_to_the_requested_directory() -> None:
    from dataset.tgb_temporal import _resolve_tgb_root_argument
    from tgb.utils.info import PROJ_DIR

    requested_root = Path.cwd() / "datasets-for-path-test"
    root_argument = _resolve_tgb_root_argument(str(requested_root))

    assert Path(PROJ_DIR + root_argument).resolve() == requested_root.resolve()
