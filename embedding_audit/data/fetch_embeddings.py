from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import argparse
import os
import pickle
import re
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path("/projects/netsi_andlab/statsbomb/megatron")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.chdir('/projects/netsi_andlab/statsbomb/megatron')
from data.statsbomb_dataset import StatsBombMatchDataset
from data.statsbomb_sequence import (
    StatsBombEventTokenizer,
    StatsBombWindowedDataset,
    windowed_collate_fn,
)

_argv = sys.argv.copy()
sys.argv = [sys.argv[0]]
from evaluate_regression import (
    build_cont_features,
    build_discrete_inputs,
    load_model,
    resolve_cont_feature_order,
)
sys.argv = _argv

from model import GPT


# -----------------------------------------------------------------------------
# League registry
# -----------------------------------------------------------------------------

LEAGUE_IDS = {
    "England - FA Women's Super League": 0,
    "England - Premier League":          1,
    "Europe - UEFA Euro":                2,
    "Germany - 1. Bundesliga":           3,
    "Germany - Frauen Bundesliga":       4,
    "Italy - Serie A":                   5,
    "Italy - Serie A Women":             6,
    "Spain - La Liga":                   7,
    "Spain - Liga F":                    8,
    "United States of America - Major League Soccer": 9,
    "United States of America - NWSL":   10,
}


def league_tag(league_name: str) -> str:
    """Short numeric tag, e.g. 'L01'."""
    lid = LEAGUE_IDS.get(league_name)
    return f"L{lid:02d}" if lid is not None else "Lxx"


def parse_team_id_from_token(team_token: str) -> Optional[int]:
    """Extract integer team id from tokens like '749:tottenham_hotspur_women'."""
    match = re.match(r"^(\d+):", team_token)
    return int(match.group(1)) if match else None


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Extract per-event layer embeddings.")

    # Paths
    p.add_argument("--checkpoint_path", default="out/statsbomb_full/ckpt_best.pt")
    p.add_argument("--index_path",      default="data/processed_fused/statsbomb_entity_active_spatial_index.json")
    p.add_argument("--meta_path",       default="data/processed_fused/meta_statsbomb_entity_active_spatial.pkl")
    p.add_argument("--data_path",       default=None)
    p.add_argument("--out_dir",         default="experiments/embedding_audit/out")

    # Match filters (mutually exclusive)
    match_filter = p.add_mutually_exclusive_group()
    match_filter.add_argument(
        "--match_indices", type=int, nargs="+", default=None,
        help="Explicit match indices to process.",
    )
    match_filter.add_argument(
        "--teams", type=int, nargs="+", default=None,
        help="Filter matches by team id.",
    )
    match_filter.add_argument(
        "--leagues", type=str, nargs="+", default=None,
        help="Filter matches by league name.",
    )

    # Token / layer filters
    p.add_argument(
        "--token_ids", type=int, nargs="+", default=None,
        help="Only extract embeddings for events with these token ids. Default: all.",
    )
    p.add_argument(
        "--layers", type=int, nargs="+", default=None,
        help="Layer indices to extract (0=input embedding, 1..N=transformer blocks). Default: all.",
    )

    # Output filename override
    p.add_argument(
        "--out_file", type=str, default=None,
        help="Override the auto-generated output filename (without directory). "
             "E.g. --out_file my_run.pkl",
    )

    # Output mode
    p.add_argument(
        "--raw", action="store_true",
        help="Save raw per-event embeddings instead of mean-pooled. Warning: very large output.",
    )

    # Loader / runtime
    p.add_argument("--batch_size",            type=int, default=16)
    p.add_argument("--max_events_per_window", type=int, default=512)
    p.add_argument("--num_workers",           type=int, default=4)
    p.add_argument("--seed",                  type=int, default=1337)
    p.add_argument("--no_pin_memory",         action="store_true")
    p.add_argument("--device",                type=str, default=None)

    return p.parse_args()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def token_ids_to_names(ids: List[int], tokenizer: Dict[str, int]) -> Dict[int, str]:
    id_to_name = {v: k for k, v in tokenizer.items()}
    return {i: id_to_name.get(i, f"<id:{i}>") for i in ids}


def resolve_match_indices(
    base_dataset: StatsBombMatchDataset,
    match_indices: Optional[List[int]],
    teams: Optional[List[int]],
    leagues: Optional[List[str]],
) -> List[int]:
    n = len(base_dataset)
    if match_indices is not None:
        return match_indices
    if teams is None and leagues is None:
        return list(range(n))

    teams_set   = set(teams) if teams else None
    leagues_set = set(l.lower() for l in leagues) if leagues else None

    selected = []
    for i in range(n):
        meta = base_dataset.get_match(i, include_tokens=False)
        if leagues_set is not None:
            if str(meta.get("league", "")).lower() in leagues_set:
                selected.append(i)
        elif teams_set is not None:
            if meta.get("home_team_id") in teams_set or meta.get("away_team_id") in teams_set:
                selected.append(i)

    print(f"Match filter selected {len(selected)} / {n} matches.")
    return selected


def make_output_filename(
    layer_indices: Optional[List[int]],
    token_id_set: Optional[Set[int]],
    leagues: Optional[List[str]],
    raw: bool,
) -> str:
    layers_str = "layers_all" if layer_indices is None else \
        "layers_" + "-".join(str(l) for l in sorted(layer_indices))
    tokens_str = "tokens_all" if token_id_set is None else \
        "tokens_" + "-".join(str(t) for t in sorted(token_id_set))
    if leagues is None:
        leagues_str = "leagues_all"
    else:
        tags = "-".join(league_tag(l) for l in sorted(leagues, key=lambda x: LEAGUE_IDS.get(x, 99)))
        leagues_str = f"leagues_{tags}"
    mode_str = "raw" if raw else "pooled"
    return f"embeddings_{mode_str}_{leagues_str}_{layers_str}_{tokens_str}.pkl"


# -----------------------------------------------------------------------------
# Forward pass + per-event extraction
# -----------------------------------------------------------------------------

def extract_window_layer_embeddings(
    model: GPT,
    batch: Dict[str, torch.Tensor],
    cont_feature_order: List[str],
    device_obj: torch.device,
    layer_indices: Optional[List[int]],
    token_id_set: Optional[Set[int]],
    team_id_vocab: Dict[int, int],  # vocab_team_id -> integer team id
) -> Dict[str, Any]:
    """
    Returns filtered embeddings with per-event team assignments.

    team_id_vocab maps the token-vocab team id (e.g. 2 for tottenham) to the
    integer team id extracted from the token string (e.g. 749).
    """
    if "team_id" not in batch:
        raise KeyError(
            "Expected 'team_id' key in batch but it was not found. "
            "Check that StatsBombWindowedDataset / windowed_collate_fn includes "
            "team_id in the returned batch dict."
        )

    input_ids     = batch["input_ids"].to(device_obj, non_blocking=True)
    attn_mask     = batch["attention_mask"].to(device_obj, non_blocking=True)
    team_ids_raw  = batch["team_id"].cpu().numpy()   # (B, T), vocab-space team ids
    cont_features = build_cont_features(batch, cont_feature_order, device_obj)
    discrete_input_heads = list(getattr(model.config, "discrete_input_heads", []) or [])
    discrete_inputs = build_discrete_inputs(batch, discrete_input_heads, device_obj)

    all_layers = model.get_layer_embeddings(
        idx=input_ids,
        attention_mask=attn_mask,
        cont_features=cont_features,
        time_values=batch["abs_time"].to(device_obj, non_blocking=True).float(),
        **discrete_inputs,
    )  # (B, T, n_layer+1, n_embd)

    if layer_indices is not None:
        all_layers = all_layers[:, :, layer_indices, :]

    all_layers  = all_layers.half().cpu().numpy()
    attn_cpu    = attn_mask.cpu().numpy().astype(bool)
    ids_cpu     = input_ids.cpu().numpy()
    match_idxs  = batch["match_idx"].cpu().tolist()

    out_embs, out_tids, out_teams, out_midxs = [], [], [], []
    for i in range(len(match_idxs)):
        valid_mask = attn_cpu[i]
        # Exclude game boundary tokens (vocab team id 0=UNK, 1=game)
        valid_mask = valid_mask & (team_ids_raw[i] > 1)
        if token_id_set is not None:
            valid_mask = valid_mask & np.isin(ids_cpu[i], list(token_id_set))
        if not valid_mask.any():
            continue

        # Map vocab-space team ids -> integer team ids
        vocab_team_ids = team_ids_raw[i][valid_mask]
        int_team_ids   = np.array(
            [team_id_vocab.get(int(v), -1) for v in vocab_team_ids], dtype=np.int32
        )

        out_embs.append(all_layers[i][valid_mask])
        out_tids.append(ids_cpu[i][valid_mask])
        out_teams.append(int_team_ids)
        out_midxs.extend([match_idxs[i]] * int(valid_mask.sum()))

    if not out_embs:
        return {"embeddings": None, "token_ids": None, "team_ids": None, "match_idxs": []}

    return {
        "embeddings": np.concatenate(out_embs,  axis=0),  # (N, L, D)
        "token_ids":  np.concatenate(out_tids,  axis=0),  # (N,)
        "team_ids":   np.concatenate(out_teams, axis=0),  # (N,)
        "match_idxs": out_midxs,
    }


# -----------------------------------------------------------------------------
# Assemble: pooled (league -> team_id -> season -> token_id -> (L, D))
# -----------------------------------------------------------------------------

def assemble_pooled(
    match_data: Dict[int, Dict],  # match_idx -> {entries, league, season}
    layers_stored: List[int],
) -> Dict:
    """
    Mean-pool embeddings per league / team / season / token / layer.

    Output structure:
        {
            "layers": [int, ...],
            "leagues": {
                "<league_name>": {
                    <team_id>: {
                        "<season>": {
                            <token_id>: np.ndarray  # (n_layers, n_embd), float16
                        }
                    }
                }
            }
        }
    """
    # Accumulate: (league, team_id, season, token_id) -> list of (L, D) arrays
    accum: Dict[Tuple, list] = defaultdict(list)

    for match_idx, md in tqdm(match_data.items(), desc="Pooling", unit="match"):
        league = md["league"]
        season = md["season"]
        for _, emb, tok_id, team_id in md["entries"]:
            if team_id < 0:
                continue
            accum[(league, int(team_id), season, int(tok_id))].append(emb)  # (L, D)

    # Build nested output dict
    output: Dict[str, Any] = {"layers": layers_stored, "leagues": defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )}

    for (league, team_id, season, tok_id), embs in accum.items():
        mean_emb = np.stack(embs, axis=0).mean(axis=0).astype(np.float16)  # (L, D)
        output["leagues"][league][team_id][season][tok_id] = mean_emb

    # Convert nested defaultdicts
    output["leagues"] = {
        league: {
            team: dict(seasons)
            for team, seasons in teams.items()
        }
        for league, teams in output["leagues"].items()
    }
    return output


# -----------------------------------------------------------------------------
# Assemble: raw (league -> team_id -> season -> match_id -> {embedding, token_ids})
# -----------------------------------------------------------------------------

def assemble_raw(
    match_data: Dict[int, Dict],
    layers_stored: List[int],
) -> Dict:
    """
    Raw per-event embeddings, organised by league / team / season / match.

    Output structure:
        {
            "layers": [int, ...],
            "leagues": {
                "<league_name>": {
                    <team_id>: {
                        "<season>": {
                            "<match_id>": {
                                "embedding": np.ndarray  # (n_events, L, D), float16
                                "token_ids": np.ndarray  # (n_events,), int32
                            }
                        }
                    }
                }
            }
        }
    """
    output: Dict[str, Any] = {"layers": layers_stored, "leagues": defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )}

    for match_idx, md in tqdm(match_data.items(), desc="Assembling", unit="match"):
        league   = md["league"]
        season   = md["season"]
        match_id = md["match_id"]

        # Split events by team
        by_team: Dict[int, Tuple[list, list]] = defaultdict(lambda: ([], []))
        for _, emb, tok_id, team_id in md["entries"]:
            if team_id < 0:
                continue
            by_team[int(team_id)][0].append(emb)
            by_team[int(team_id)][1].append(tok_id)

        for team_id, (embs, tids) in by_team.items():
            output["leagues"][league][team_id][season][match_id] = {
                "embedding": np.stack(embs, axis=0).astype(np.float16),
                "token_ids": np.array(tids, dtype=np.int32),
            }

    output["leagues"] = {
        league: {
            team: dict(seasons)
            for team, seasons in teams.items()
        }
        for league, teams in output["leagues"].items()
    }
    return output


# -----------------------------------------------------------------------------
# Main extraction loop
# -----------------------------------------------------------------------------

def extract_and_save(
    model: GPT,
    data_loader: DataLoader,
    base_dataset: StatsBombMatchDataset,
    cont_feature_order: List[str],
    output_dir: Path,
    device_obj: torch.device,
    layer_indices: Optional[List[int]],
    token_id_set: Optional[Set[int]],
    leagues_filter: Optional[List[str]],
    raw: bool,
    team_id_vocab: Dict[int, int],
    filename: str,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # match_idx -> {entries: [(order, emb, tok_id, team_id)], league, season, match_id}
    match_data: Dict[int, Dict] = {}
    global_order = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting embeddings", unit="batch"):
    
            result = extract_window_layer_embeddings(
                model, batch, cont_feature_order, device_obj,
                layer_indices, token_id_set, team_id_vocab,
            )
            batch_match_idxs = batch["match_idx"].cpu().tolist()

            if result["embeddings"] is not None:
                embs  = result["embeddings"]
                tids  = result["token_ids"]
                teams = result["team_ids"]
                midxs = result["match_idxs"]

                offset = 0
                seen = {}
                for j, midx in enumerate(midxs):
                    if midx not in seen:
                        seen[midx] = global_order + j
                    if offset < len(embs):
                        if midx not in match_data:
                            meta = base_dataset.get_match(int(midx), include_tokens=False)
                            match_data[int(midx)] = {
                                "entries":  [],
                                "league":   str(meta.get("league", "unknown")),
                                "season":   str(meta.get("season", "unknown")),
                                "match_id": str(meta.get("match_id", midx)),
                            }
                        match_data[int(midx)]["entries"].append(
                            (seen[midx], embs[offset], int(tids[offset]), int(teams[offset]))
                        )
                        offset += 1

            global_order += len(batch_match_idxs)

    # Sort entries within each match by order
    for md in match_data.values():
        md["entries"].sort(key=lambda x: x[0])

    layers_stored = layer_indices if layer_indices is not None else list(range(model.config.n_layer + 1))
    output = assemble_raw(match_data, layers_stored) if raw else assemble_pooled(match_data, layers_stored)

    out_path = output_dir / filename
    print(f"Saving to {out_path} ...")
    with out_path.open("wb") as fh:
        pickle.dump(output, fh)
    print("Done.")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_obj = torch.device(device)

    ckpt = Path(args.checkpoint_path)
    if not ckpt.exists():
        print(f"Checkpoint not found: {ckpt}")
        sys.exit(1)

    model, checkpoint_config = load_model(ckpt, device_obj)
    cont_feature_order = resolve_cont_feature_order(checkpoint_config, model.config.num_cont_features)

    base_dataset = StatsBombMatchDataset(
        data_path=args.data_path,
        meta_path=args.meta_path,
        index_path=args.index_path,
        map_location="cpu",
    )
    tokenizer = StatsBombEventTokenizer.from_dataset(base_dataset)

    # Build vocab-space team id -> integer team id mapping from meta
    with open(args.meta_path, "rb") as f:
        meta = pickle.load(f)
    team_vocab = meta["teams"]["token_to_id"]  # token string -> vocab id
    team_id_vocab: Dict[int, int] = {}
    for token_str, vocab_id in team_vocab.items():
        int_id = parse_team_id_from_token(token_str)
        if int_id is not None:
            team_id_vocab[vocab_id] = int_id

    match_indices = resolve_match_indices(
        base_dataset,
        match_indices=args.match_indices,
        teams=args.teams,
        leagues=args.leagues,
    )
    if not match_indices:
        print("No matches selected — check your filter arguments.")
        sys.exit(1)

    token_id_set = set(args.token_ids) if args.token_ids is not None else None
    if token_id_set is not None:
        tok = tokenizer.token_to_id if hasattr(tokenizer, "token_to_id") else tokenizer
        print(f"Token filter: {token_ids_to_names(list(token_id_set), tok)}")

    n_layers_total = model.config.n_layer + 1
    if args.layers is not None:
        invalid = [l for l in args.layers if l < 0 or l >= n_layers_total]
        if invalid:
            print(f"[WARN] Layer indices out of range (0..{n_layers_total - 1}): {invalid}")
        layer_indices = [l for l in args.layers if 0 <= l < n_layers_total]
    else:
        layer_indices = None

    print(f"Layers: {'all' if layer_indices is None else layer_indices}")
    print(f"Output mode: {'raw' if args.raw else 'mean-pooled'}")

    windowed_dataset = StatsBombWindowedDataset(
        base_dataset=base_dataset,
        tokenizer=tokenizer,
        match_indices=match_indices,
        max_events_per_window=args.max_events_per_window,
        chunking="fixed",
        fixed_chunk_offset=0,
        include_segment_start_chunk=True,
    )

    loader = DataLoader(
        windowed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        persistent_workers=args.num_workers > 0,
        collate_fn=windowed_collate_fn,
    )

    filename = args.out_file if args.out_file is not None else \
        make_output_filename(layer_indices, token_id_set, args.leagues, args.raw)

    extract_and_save(
        model, loader, base_dataset, cont_feature_order,
        Path(args.out_dir), device_obj,
        layer_indices, token_id_set, args.leagues, args.raw,
        team_id_vocab, filename,
    )


if __name__ == "__main__":
    main()