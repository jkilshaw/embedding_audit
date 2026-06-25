from pathlib import Path

import pytest

from embedding_audit.data.load_embeddings import (
    load_checkpoint,
    load_meta,
    load_team_embeddings,
)


CHECKPOINT_PATH = "/projects/netsi_andlab/statsbomb/megatron/out/statsbomb_half_end/ckpt_best.pt"
META_PATH = "/projects/netsi_andlab/statsbomb/megatron/data/processed_fused/meta_statsbomb_entity_active_spatial.pkl"


def test_load_team_embeddings_with_real_cluster_files():
    checkpoint_path = CHECKPOINT_PATH
    meta_path = META_PATH
    if not Path(checkpoint_path).exists() or not Path(meta_path).exists():
        pytest.skip("Cluster checkpoint/meta files are not available.")

    ckpt = load_checkpoint(checkpoint_path)
    meta = load_meta(meta_path)
    team_embeds = load_team_embeddings(checkpoint_path, meta_path)

    team_weight = ckpt["model"]["team_embed.weight"]
    team_token_to_id = meta["teams"]["token_to_id"]

    assert team_weight.ndim == 2
    assert len(team_token_to_id) > 0
    assert len(team_embeds) > 0
    assert all(isinstance(team_id, int) for team_id in team_embeds)
    assert all(vector.ndim == 1 for vector in team_embeds.values())
    assert all(vector.shape[0] == team_weight.shape[1] for vector in team_embeds.values())

    print("team_weight shape:", tuple(team_weight.shape))
    print("num metadata team tokens:", len(team_token_to_id))
    print("num loaded team embeddings:", len(team_embeds))
    print("first 10 token_to_id entries:", list(team_token_to_id.items())[:10])
    print(
        "first 10 loaded team embeddings:",
        [
            (team_id, vector.shape, vector[:5].tolist())
            for team_id, vector in list(team_embeds.items())[:10]
        ],
    )
