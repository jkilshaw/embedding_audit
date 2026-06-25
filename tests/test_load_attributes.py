import json
from pathlib import Path

import pytest

from embedding_audit.data.load_attributes import (
    league_to_sex,
    load_q1_attributes,
    load_team_attributes,
    make_league_dict,
)
from embedding_audit.data.load_embeddings import load_team_embeddings


CHECKPOINT_PATH = "/projects/netsi_andlab/statsbomb/megatron/out/statsbomb_half_end/ckpt_best.pt"
META_PATH = "/projects/netsi_andlab/statsbomb/megatron/data/processed_fused/meta_statsbomb_entity_active_spatial.pkl"
INDEX_PATH = "/projects/netsi_andlab/statsbomb/megatron/data/processed_fused/statsbomb_entity_active_spatial_index.json"


def write_index(tmp_path, matches):
    index_path = tmp_path / "statsbomb_entity_active_spatial_index.json"
    index_path.write_text(json.dumps({"matches": matches}))
    return index_path


def test_make_league_dict_uses_first_match_for_requested_team_ids():
    matches = [
        {
            "league": "England - FA Women's Super League",
            "home_team_id": 749,
            "away_team_id": 851,
        },
        {
            "league": "Spain - Liga F",
            "home_team_id": 749,
            "away_team_id": 900,
        },
        {
            "league": "England - Premier League",
            "home_team_id": 1,
            "away_team_id": 2,
        },
    ]

    team_to_league = make_league_dict([749, 900, 12345], matches)

    assert team_to_league == {
        749: "England - FA Women's Super League",
        900: "Spain - Liga F",
    }


def test_league_to_sex_matches_q1_audit_rule():
    assert league_to_sex("Spain - Liga F") == "women"
    assert league_to_sex("England - Premier League") == "men"


def test_load_team_attributes_returns_league_and_sex_for_team_ids(tmp_path):
    index_path = write_index(
        tmp_path,
        [
            {
                "league": "England - FA Women's Super League",
                "home_team_id": 749,
                "away_team_id": 851,
            },
            {
                "league": "England - Premier League",
                "home_team_id": 1,
                "away_team_id": 2,
            },
        ],
    )

    attributes = load_team_attributes(str(index_path), [749, 851, 1, 999])

    assert set(attributes) == {"league", "sex"}
    assert attributes["league"] == {
        749: "England - FA Women's Super League",
        851: "England - FA Women's Super League",
        1: "England - Premier League",
    }
    assert attributes["sex"] == {
        749: "women",
        851: "women",
        1: "men",
    }


def test_load_q1_attributes_dispatches_team_attributes(tmp_path):
    index_path = write_index(
        tmp_path,
        [
            {
                "league": "United States of America - NWSL",
                "home_team_id": 101,
                "away_team_id": 102,
            },
        ],
    )

    attributes = load_q1_attributes("team", str(index_path), [101, 102])

    assert attributes["league"] == {
        101: "United States of America - NWSL",
        102: "United States of America - NWSL",
    }
    assert attributes["sex"] == {101: "women", 102: "women"}


def test_load_q1_attributes_rejects_unimplemented_embedding_type(tmp_path):
    index_path = write_index(tmp_path, [])

    with pytest.raises(NotImplementedError, match="player"):
        load_q1_attributes("player", str(index_path), [])


def test_load_team_attributes_with_real_cluster_embedding_ids():
    missing_paths = [path for path in (CHECKPOINT_PATH, META_PATH, INDEX_PATH)
                     if not Path(path).exists()]
    if missing_paths:
        pytest.skip(f"Cluster data files are not available: {missing_paths}")

    team_embeds = load_team_embeddings(CHECKPOINT_PATH, META_PATH)
    attributes = load_q1_attributes("team", INDEX_PATH, team_embeds.keys())

    print("num loaded team embeddings:", len(team_embeds))
    print(
        "first 10 team embeddings:",
        [
            (team_id, vector.shape, vector[:5].tolist())
            for team_id, vector in list(team_embeds.items())[:10]
        ],
    )
    print("attribute names:", list(attributes))
    print("num league attributes:", len(attributes["league"]))
    print("num sex attributes:", len(attributes["sex"]))
    print("first 10 league attributes:", list(attributes["league"].items())[:10])
    print("first 10 sex attributes:", list(attributes["sex"].items())[:10])
    aligned_team_ids = [
        team_id
        for team_id in team_embeds
        if team_id in attributes["league"] and team_id in attributes["sex"]
    ][:5]
    print(
        "first 5 aligned embedding/attribute rows:",
        [
            {
                "sample_index": sample_index,
                "team_id": team_id,
                "league": attributes["league"][team_id],
                "sex": attributes["sex"][team_id],
                "embedding_shape": team_embeds[team_id].shape,
                "embedding_first_5": team_embeds[team_id][:5].tolist(),
            }
            for sample_index, team_id in enumerate(aligned_team_ids)
        ],
    )

    assert set(attributes) == {"league", "sex"}
    assert len(team_embeds) > 0
    assert len(attributes["league"]) > 0
    assert len(attributes["sex"]) == len(attributes["league"])
    assert set(attributes["league"]).issubset(team_embeds)
    assert set(attributes["sex"]).issubset(team_embeds)
    assert all(isinstance(team_id, int) for team_id in attributes["league"])
    assert all(isinstance(league, str) for league in attributes["league"].values())
    assert set(attributes["sex"].values()).issubset({"men", "women"})
