"""
Embedding Audit Script
======================
Probes team, team-season, combined, and player embeddings for interpretable
structure using logistic regression, PCA component probes, PLS, and block
coherence.  Handles ablated models where one or more embedding types may be absent.

Four research questions:
  Q1. Does the *team* embedding encode sex, league, or both?
  Q2. Does the *team-season* embedding encode team, league, season, or some combination?
  Q3. Does the *combined* (team + team-season) embedding encode league or season or both?
  Q4. Does the *player* embedding encode team, position_group, position, or some combination?

Prior findings that guide interpretation:
  - Team embedding encodes both sex and league, hierarchically.
  - Team-season embedding encodes season but not team (within a league).
  - Player embedding encodes position group.
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os, json, pickle, textwrap, warnings, argparse
from pathlib import Path
from collections import defaultdict
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tools import (
    load_pickle,
    make_league_dict,
    cos_sim_mat,
    pca,
    umap_embed,
    plot_heatmap,
    plot_pca,
    plot_umap,
    block_coherence,
    build_team_season_dict,
    build_ts_embeds,
    league_probe,
    league_component_probe,
    league_pls_probe,
    get_primary_affiliation,
    league_to_sex,
    _make_attribute_colormap,
)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# =============================================================================
# CONFIG
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Embedding Audit Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Paths ---
    g = p.add_argument_group("paths")
    g.add_argument("--checkpoint",    default="statsbomb_full/ckpt_best.pt",
                   metavar="PATH",    help="Model checkpoint (.pt)")
    g.add_argument("--meta",          default="meta_statsbomb_entity_active_spatial.pkl",
                   metavar="PATH",    help="Entity metadata pickle")
    g.add_argument("--index",         default="statsbomb_entity_active_spatial_index.json",
                   metavar="PATH",    help="Entity index JSON")
    g.add_argument("--team-season-meta", default="statsbomb_entity_active_spatial_team_season_mapping.pkl",
                   metavar="PATH",    help="Team-season mapping pickle")
    g.add_argument("--output-dir",    default="audit_output",
                   metavar="DIR",     help="Directory for all outputs")

    # --- Embedding keys (pass 'none' to mark as absent / ablated) ---
    g = p.add_argument_group("embedding keys  (pass 'none' if ablated)")
    g.add_argument("--team-key",        default="team_embed",
                   metavar="KEY",       help="Embedding table key for team")
    g.add_argument("--team-season-key", default="team_season_embed",
                   metavar="KEY",       help="Embedding table key for team-season")
    g.add_argument("--player-key",      default="player_embed",
                   metavar="KEY",       help="Embedding table key for player")

    # --- Analysis parameters ---
    g = p.add_argument_group("analysis parameters")
    g.add_argument("--player-sample-n",   type=int,   default=2000,
                   metavar="N",           help="Players to subsample (proportional by league)")
    g.add_argument("--min-samples-class", type=int,   default=5,
                   metavar="N",           help="Minimum samples per class for CV logistic regression")
    g.add_argument("--n-folds",           type=int,   default=5,
                   metavar="K",           help="Stratified k-fold count for logistic regression")
    g.add_argument("--umap-accuracy-gate",type=float, default=0.80,
                   metavar="THRESH",      help="Only run UMAP when CV-LR accuracy exceeds this")
    g.add_argument("--trajectory-highlight", nargs="+",
                   default=["manchester_city", "liverpool", "arsenal", "leicester_city"],
                   metavar="TEAM",        help="Teams to highlight in trajectory plots")
    g.add_argument("--seed",             type=int,   default=1337,
                   metavar="N",           help="Global random seed")
    g.add_argument("--label-umap",       action="store_true", default=False,
                   help="Label points on team-related UMAPs (not applied to player UMAPs)")

    return p.parse_args()


def _none_if_str(val):
    """Convert the string 'none' (case-insensitive) to Python None."""
    return None if isinstance(val, str) and val.lower() == "none" else val


args = _parse_args()

# --- Paths ---
CHECKPOINT_PATH  = "/projects/netsi_andlab/statsbomb/megatron/out/" + args.checkpoint
META_PATH        = "/projects/netsi_andlab/statsbomb/megatron/data/processed_fused/" + args.meta
INDEX_PATH       = "/projects/netsi_andlab/statsbomb/megatron/data/processed_fused/" + args.index
TEAM_SEASON_META = "/projects/netsi_andlab/statsbomb/megatron/data/processed_fused/" + args.team_season_meta
OUTPUT_DIR       = "/projects/netsi_andlab/statsbomb/megatron/experiments/embedding_audit/out/" + args.output_dir

# --- Embedding keys ---
TEAM_KEY         = _none_if_str(args.team_key)
TEAM_SEASON_KEY  = _none_if_str(args.team_season_key)
PLAYER_KEY       = _none_if_str(args.player_key)

# --- Analysis parameters ---
PLAYER_SAMPLE_N      = args.player_sample_n
MIN_SAMPLES_CLASS    = args.min_samples_class
N_FOLDS              = args.n_folds
UMAP_ACCURACY_GATE   = args.umap_accuracy_gate
TRAJECTORY_HIGHLIGHT = args.trajectory_highlight
RANDOM_SEED          = args.seed
LABEL_UMAP           = args.label_umap

# =============================================================================
# OUTPUT SETUP
# =============================================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
_plot_counter = [0]

def save_fig(fig, name):
    _plot_counter[0] += 1
    path = os.path.join(OUTPUT_DIR, f"{_plot_counter[0]:02d}_{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _banner(text):
    line = "=" * 70
    print(f"\n{line}\n  {text}\n{line}")


def _subhead(text):
    print(f"\n--- {text} ---")

# =============================================================================
# DATA LOADING
# =============================================================================
_banner("Loading data")

import sys
sys.path.insert(0, "/projects/netsi_andlab/statsbomb/megatron/")  # adjust to wherever model.py lives
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
with open(META_PATH, "rb") as f:
    meta = pickle.load(f)
with open(INDEX_PATH, "r") as f:
    data_dict = json.load(f)

team_token_to_id   = meta["teams"]["token_to_id"]
player_token_to_id = meta["players"]["token_to_id"]

player_to_team      = load_pickle("/projects/netsi_andlab/statsbomb/megatron/experiments/embedding_audit/data/player_to_team.pkl")
player_to_position  = load_pickle("/projects/netsi_andlab/statsbomb/megatron/experiments/embedding_audit/data/player_to_position.pkl")
position_categories = load_pickle("/projects/netsi_andlab/statsbomb/megatron/experiments/embedding_audit/data/position_categories.pkl")

# ---- Team embeddings --------------------------------------------------------
team_weight        = None
team_embedding_dict = {}   # int team_id → tensor/array
team_embeddings_raw = {}   # "749:tottenham_hotspur_women" → tensor

if TEAM_KEY is not None:
    key = f"{TEAM_KEY}.weight"
    if key not in ckpt["model"]:
        print(f"[WARN] '{key}' not in checkpoint — treating TEAM_KEY as absent.")
        TEAM_KEY = None
    else:
        team_weight = ckpt["model"][key]
        for name, idx in team_token_to_id.items():
            team_embeddings_raw[name] = team_weight[idx]
        team_embedding_dict = {
            int(k.split(":")[0]): v
            for k, v in team_embeddings_raw.items()
            if ":" in k
        }
        print(f"  Team embeddings: {len(team_embedding_dict)} teams loaded.")

# Build id → human-readable name for all teams regardless of whether
# team embeddings are present, so Q1 UMAP labels work correctly.
team_id_to_name = {
    int(k.split(":")[0]): k.split(":", 1)[1]
    for k in team_token_to_id
    if ":" in k
}

# ---- Team-season embeddings -------------------------------------------------
team_season_weight = None
team_season_id_meta = None
team_season_dict    = {}

if TEAM_SEASON_KEY is not None:
    key = f"{TEAM_SEASON_KEY}.weight"
    if key not in ckpt["model"]:
        print(f"[WARN] '{key}' not in checkpoint — treating TEAM_SEASON_KEY as absent.")
        TEAM_SEASON_KEY = None
    else:
        team_season_weight  = ckpt["model"][key]
        team_season_id_meta = load_pickle(TEAM_SEASON_META)
        team_season_dict    = build_team_season_dict(team_season_id_meta, team_season_weight)
        n_ts = sum(len(v["seasons"]) for v in team_season_dict.values())
        print(f"  Team-season embeddings: {len(team_season_dict)} teams, {n_ts} team-seasons loaded.")

# ---- Player embeddings ------------------------------------------------------
player_weight = None
player_embeddings = {}  # int player_id → tensor

if PLAYER_KEY is not None:
    key = f"{PLAYER_KEY}.weight"
    if key not in ckpt["model"]:
        print(f"[WARN] '{key}' not in checkpoint — treating PLAYER_KEY as absent.")
        PLAYER_KEY = None
    else:
        player_weight = ckpt["model"][key]
        for name, idx in player_token_to_id.items():
            player_embeddings[idx] = player_weight[idx]
        print(f"  Player embeddings: {len(player_embeddings)} players loaded.")

# ---- League / sex attribute dicts -------------------------------------------
# Use all teams from token_to_id, not just those with embeddings,
# so player league lookups work even when TEAM_KEY is ablated.
matches = data_dict["matches"]
sex_identifiers = ["Women", "Frauen", "Liga F", "NWSL"]

all_team_ids = [
    int(k.split(":")[0])
    for k in team_token_to_id
    if ":" in k
]
team_to_league = make_league_dict(all_team_ids, matches)

team_to_sex = {
    tid: ("women" if any(s in team_to_league[tid] for s in sex_identifiers) else "men")
    for tid in all_team_ids
    if tid in team_to_league
}

# fake_to_real: shard int → canonical int team id
fake_to_real_tid = {v: int(k.split(":")[0]) for k, v in team_token_to_id.items() if v > 1}

# ---- Player attribute dicts -------------------------------------------------
player_to_pos_group = {}
for player in player_to_position:
    if player in (0, 1):
        continue
    player_to_pos_group[player] = get_primary_affiliation(
        player, player_to_position, position_cats=position_categories
    )

player_primary_team = {
    p: get_primary_affiliation(p, player_to_team)
    for p in player_to_pos_group
}

# Build league / sex attrs for players via their primary team
def _player_league(pid):
    team_shard_id = player_primary_team.get(pid)
    if team_shard_id is None:
        return None
    real_tid = fake_to_real_tid.get(team_shard_id)
    if real_tid is None:
        return None
    return team_to_league.get(real_tid)

# Valid players: present in embeddings AND have position + team info
player_ids_full = [
    p for p in player_embeddings
    if p in player_primary_team
    and p in player_to_pos_group
    and _player_league(p) is not None
]

player_league_attr = {p: _player_league(p) for p in player_ids_full}
player_sex_attr    = {p: league_to_sex(player_league_attr[p]) for p in player_ids_full}

# ── Block 1: replace the raw_ids debug block (before sampling) ───────────────
raw_ids = list(player_embeddings.keys())[:5]
print("Raw embed IDs:", raw_ids)
for pid in raw_ids:
    real_tid = fake_to_real_tid.get(player_primary_team.get(pid))
    print(f"  pid={pid}  primary_team_shard={player_primary_team.get(pid)}  "
          f"real_tid={real_tid}  league={team_to_league.get(real_tid)}  "
          f"pos_group={player_to_pos_group.get(pid)}")

# Subsample proportionally by league
random.seed(RANDOM_SEED)
by_league = defaultdict(list)
for p in player_ids_full:
    by_league[player_league_attr[p]].append(p)

sample = []
for league, players in by_league.items():
    n = max(1, round(PLAYER_SAMPLE_N * len(players) / len(player_ids_full)))
    sample.extend(random.sample(players, min(n, len(players))))
sample = sample[:PLAYER_SAMPLE_N]

player_sample_embeds    = {p: player_embeddings[p] for p in sample}
player_sample_team      = {p: str(player_primary_team[p]) for p in sample}
player_sample_league    = {p: player_league_attr[p] for p in sample}
player_sample_sex       = {p: player_sex_attr[p] for p in sample}
player_sample_pos_group = {p: player_to_pos_group[p] for p in sample}

# ── Block 2: replace the post-sample debug block ─────────────────────────────
print("player_ids_full:", len(player_ids_full))
print("Sample size:", len(sample))
print("player_sample_embeds size:", len(player_sample_embeds))
print("Example embed keys  (first 5):", list(player_sample_embeds.keys())[:5])
print("Example sex keys    (first 5):", list(player_sample_sex.keys())[:5])
print("by_league counts:", {lg: len(ps) for lg, ps in by_league.items()})

print(f"  Player sample: {len(sample)} players across {len(by_league)} leagues.")

# =============================================================================
# GENERIC PROBE RUNNER
# =============================================================================

def probe_attribute(
    embeds_dict,
    attr_dict,
    attr_name,
    embed_name,
    min_samples=MIN_SAMPLES_CLASS,
    umap_gate=UMAP_ACCURACY_GATE,
    n_folds=N_FOLDS,
    max_classes_for_heatmap=60,
    pca_variance_threshold=0.95,
    output_dir=OUTPUT_DIR,
    skip_pca=False,
    also_euclidean_bc=False,
    label_points=False,
    id_to_name=None,
):
    """
    Run LR / PCA-component / PLS / block-coherence probes for one
    (embedding, attribute) pair.  Returns a results dict and generates plots.

    Parameters
    ----------
    embeds_dict  : dict  id → np.ndarray
    attr_dict    : dict  id → label string
    attr_name    : str   human-readable attribute name  (e.g. "league")
    embed_name   : str   human-readable embedding name  (e.g. "team")
    label_points : bool  annotate each point with its id on UMAP plots
    id_to_name   : dict  optional id → display string used for UMAP point labels
    """
    tag = f"{embed_name}→{attr_name}"
    _subhead(f"Probing {tag}")

    # --- Align ids and filter small classes ----------------------------------
    ids = [i for i in embeds_dict if i in attr_dict and attr_dict[i] is not None]
    if not ids:
        print(f"  [SKIP] No overlap between embeds_dict and attr_dict for {tag}.")
        return None

    label_counts = defaultdict(int)
    for i in ids:
        label_counts[attr_dict[i]] += 1
    valid_labels = {lbl for lbl, cnt in label_counts.items() if cnt >= min_samples}
    ids = [i for i in ids if attr_dict[i] in valid_labels]

    if len(valid_labels) < 2:
        print(f"  [SKIP] Fewer than 2 classes with >= {min_samples} samples for {tag}.")
        return None

    n_classes = len(valid_labels)
    n_items   = len(ids)
    chance    = 1.0 / n_classes

    filtered_embeds = {i: embeds_dict[i] for i in ids}
    filtered_attrs  = {i: attr_dict[i]   for i in ids}

    print(f"  n={n_items}, n_classes={n_classes}, chance={chance:.2%}")

    results = {
        "embed_name": embed_name,
        "attr_name":  attr_name,
        "n_items":    n_items,
        "n_classes":  n_classes,
        "chance":     chance,
    }

    # =========================================================================
    # 1. LOGISTIC REGRESSION PROBE
    # =========================================================================
    try:
        lr_res = league_probe(
            filtered_embeds, filtered_attrs,
            n_folds=n_folds, random_state=RANDOM_SEED
        )
        acc = lr_res["overall_accuracy"]
        results["lr_accuracy"] = acc
        results["lr_fold_accs"] = lr_res["cv_fold_accuracies"]
        results["lr_report"] = lr_res["classification_report"]
        results["lr_per_class"] = lr_res["per_league_accuracy"]
        print(f"  LR accuracy:  {acc:.3f}  (chance {chance:.3f},  "
              f"lift {acc/chance:.1f}x)")
    except Exception as e:
        print(f"  [WARN] LR failed: {e}")
        lr_res = None
        results["lr_accuracy"] = None

    # =========================================================================
    # 2. BLOCK COHERENCE (cosine)
    # =========================================================================
    try:
        sim_mat, sorted_ids, sorted_attrs = cos_sim_mat(
            filtered_embeds, attribute_dict=filtered_attrs
        )
        bc = block_coherence(sim_mat, sorted_attrs)
        results["block_coherence"] = bc
        print(f"  Block coherence (cosine): {bc:+.4f}  "
              f"(positive = within-group > cross-group)")

        if also_euclidean_bc:
            bc_euc = _block_coherence_euclidean(filtered_embeds, filtered_attrs)
            results["block_coherence_euclidean"] = bc_euc
            print(f"  Block coherence (euclidean): {bc_euc:+.4f}  "
                  f"(positive = cross-group dist > within-group dist)")

        # Plot heatmap only if n_items is tractable
        if n_items <= max_classes_for_heatmap:
            fig = plot_heatmap(
                sim_mat, sorted_ids, sorted_attrs,
                title=f"Cosine sim heatmap: {embed_name} sorted by {attr_name}",
                figsize=(10, 8),
            )
            save_fig(fig, f"heatmap_{embed_name}_{attr_name}")
    except Exception as e:
        print(f"  [WARN] Block coherence failed: {e}")
        results["block_coherence"] = None

    # =========================================================================
    # 3. PCA COMPONENT PROBE
    # =========================================================================
    if skip_pca:
        results["pca_n_significant"] = None
        results["pca_n_components"]  = None
    else:
        try:
            pca_res = league_component_probe(
                filtered_embeds, filtered_attrs,
                variance_threshold=pca_variance_threshold,
                random_state=RANDOM_SEED,
            )
            sig_pca   = pca_res["significant_components"]
            n_sig_pca = len(sig_pca)
            n_used    = pca_res["n_components_used"]
            results["pca_n_significant"] = n_sig_pca
            results["pca_n_components"]  = n_used
            results["pca_results_table"] = pca_res["results_table"]
            print(f"  PCA: {n_sig_pca}/{n_used} components significantly encode {attr_name} (BH-corrected)")
            if sig_pca:
                evars = pca_res["explained_variance"][sig_pca]
                print(f"       Significant PCs: {sig_pca[:10]}  "
                      f"(explain {evars.sum():.1%} var)")

            fig = _plot_component_significance(
                pca_res["results_table"],
                title=f"PCA components encoding {attr_name} | {embed_name}",
                x_col="explained_var",
                x_label="Explained variance",
            )
            save_fig(fig, f"pca_sig_{embed_name}_{attr_name}")

        except Exception as e:
            print(f"  [WARN] PCA component probe failed: {e}")
            results["pca_n_significant"] = None

    # =========================================================================
    # 4. PLS PROBE
    # =========================================================================
    try:
        pls_res = league_pls_probe(
            filtered_embeds, filtered_attrs,
            random_state=RANDOM_SEED,
        )
        sig_pls   = pls_res["significant_components"]
        n_sig_pls = len(sig_pls)
        n_used_pls = pls_res["n_components_used"]
        results["pls_n_significant"] = n_sig_pls
        results["pls_n_components"]  = n_used_pls
        results["pls_results_table"] = pls_res["results_table"]
        print(f"  PLS: {n_sig_pls}/{n_used_pls} components significantly encode {attr_name} (BH-corrected)")

        # Plot: which PLS components are significant?
        fig = _plot_component_significance(
            pls_res["results_table"],
            title=f"PLS components encoding {attr_name} | {embed_name}",
            x_col=None,
            x_label="Component index",
        )
        save_fig(fig, f"pls_sig_{embed_name}_{attr_name}")

    except Exception as e:
        print(f"  [WARN] PLS probe failed: {e}")
        pls_res = None
        results["pls_n_significant"] = None

    # =========================================================================
    # 5. UMAP — only if LR accuracy exceeds gate
    # =========================================================================
    lr_acc = results.get("lr_accuracy") or 0.0
    if lr_acc >= umap_gate:
        print(f"  LR accuracy {lr_acc:.2%} ≥ gate {umap_gate:.2%} → running UMAP …")
        try:
            import traceback as _tb
            u_coords, u_ids, _ = umap_embed(filtered_embeds, random_state=RANDOM_SEED)

            fig = plot_umap(
                u_coords, u_ids,
                attribute_dict=filtered_attrs,
                title=f"UMAP: {embed_name} coloured by {attr_name} "
                      f"(LR acc {lr_acc:.2%})",
            )

            # Annotate points with names if requested.  We do this after
            # plot_umap returns so we don't rely on it accepting extra kwargs.
            if label_points:
                ax = fig.axes[0]
                for (x, y), uid in zip(u_coords, u_ids):
                    name = id_to_name[uid] if (id_to_name and uid in id_to_name) else str(uid)
                    ax.annotate(
                        name, xy=(x, y),
                        fontsize=5, alpha=0.7,
                        xytext=(2, 2), textcoords="offset points",
                    )
            save_fig(fig, f"umap_{embed_name}_{attr_name}")
            results["umap_run"] = True
        except Exception as e:
            print(f"  [WARN] UMAP failed: {e}")
            _tb.print_exc()
            results["umap_run"] = False
    else:
        results["umap_run"] = False

    return results


def _plot_component_significance(results_table, title, x_col, x_label):
    """
    Bar chart: p-value (BH-corrected) per component, coloured by significance.
    Optionally overlays explained variance as a line.
    """
    df = results_table.copy()
    n  = len(df)
    xs = np.arange(n)

    fig, ax1 = plt.subplots(figsize=(max(6, n * 0.35 + 2), 4))

    colours = ["#d62728" if sig else "#aec7e8"
               for sig in df["significant"]]
    ax1.bar(xs, -np.log10(df["p_value_corrected"].clip(1e-300)),
            color=colours, alpha=0.85)
    ax1.axhline(-np.log10(0.05), color="black", linestyle="--",
                linewidth=0.8, label="p=0.05")
    ax1.set_ylabel("−log₁₀(p corrected)", fontsize=9)
    ax1.set_xlabel("Component index", fontsize=9)

    if x_col in df.columns:
        ax2 = ax1.twinx()
        ax2.plot(xs, df[x_col], color="grey", linewidth=1.5,
                 marker="o", markersize=3, alpha=0.7, label=x_label)
        ax2.set_ylabel(x_label, fontsize=9, color="grey")
        ax2.tick_params(axis="y", colors="grey")

    sig_patch  = mpatches.Patch(color="#d62728", alpha=0.85, label="Significant (BH)")
    ns_patch   = mpatches.Patch(color="#aec7e8", alpha=0.85, label="Not significant")
    ax1.legend(handles=[sig_patch, ns_patch], fontsize=7, loc="upper right")
    ax1.set_title(title, fontsize=10)

    # Show only every 5th x-tick if large
    if n > 20:
        ax1.set_xticks(xs[::5])
        ax1.set_xticklabels(xs[::5])
    else:
        ax1.set_xticks(xs)
        ax1.set_xticklabels(xs)

    fig.tight_layout()
    return fig


def plot_lr_summary(all_results, title="LR Accuracy Summary"):
    """
    Horizontal bar chart of LR accuracies for all (embed, attribute) pairs,
    with chance levels shown as markers.
    """
    records = [r for r in all_results if r is not None and r.get("lr_accuracy") is not None]
    if not records:
        return None

    records.sort(key=lambda r: r["lr_accuracy"], reverse=True)
    labels  = [f"{r['embed_name']}\n→ {r['attr_name']}" for r in records]
    accs    = [r["lr_accuracy"] for r in records]
    chances = [r["chance"] for r in records]
    n = len(records)

    fig, ax = plt.subplots(figsize=(7, max(3, n * 0.45 + 1)))
    ys = np.arange(n)
    bars = ax.barh(ys, accs, height=0.6, color="steelblue", alpha=0.8)
    ax.scatter(chances, ys, color="crimson", zorder=3, s=50,
               marker="|", linewidths=2, label="Chance")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Cross-validated accuracy", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=8)
    ax.set_title(title, fontsize=11)

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{acc:.2%}", va="center", fontsize=8)

    fig.tight_layout()
    return fig


# =============================================================================
# HELPER: EUCLIDEAN BLOCK COHERENCE
# =============================================================================

def _block_coherence_euclidean(embeds_dict, attr_dict):
    """
    Block coherence using euclidean distance.
    Returns mean(cross-group distance) - mean(within-group distance).
    Positive = cross-group points are farther apart than within-group (coherent).
    """
    ids   = [i for i in embeds_dict if attr_dict.get(i) is not None]
    attrs = np.array([attr_dict[i] for i in ids])
    vecs  = np.stack([np.array(embeds_dict[i], dtype=np.float32) for i in ids])

    # pairwise euclidean distances
    diff  = vecs[:, None, :] - vecs[None, :, :]   # (N, N, D)
    dists = np.sqrt((diff ** 2).sum(-1))           # (N, N)

    within, cross = [], []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if attrs[i] == attrs[j]:
                within.append(dists[i, j])
            else:
                cross.append(dists[i, j])
    return float(np.mean(cross) - np.mean(within))


# =============================================================================
# HELPER: TOP SIMILAR PAIRS
# =============================================================================

def top_similar_pairs(embeds_dict, n=10, id_to_name=None):
    """
    Find the top-N most similar pairs by cosine similarity and by euclidean
    proximity (smallest distance).

    Parameters
    ----------
    embeds_dict : dict  id → array
    n           : int   number of pairs to return for each metric
    id_to_name  : dict  optional id → display string

    Returns
    -------
    cosine_top : list of dicts  {id_a, id_b, name_a, name_b, cosine_sim}
    eucl_top   : list of dicts  {id_a, id_b, name_a, name_b, euclidean_dist}
    """
    ids  = list(embeds_dict.keys())
    vecs = np.stack([np.array(embeds_dict[i], dtype=np.float32) for i in ids])

    # Cosine similarity
    norms    = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)
    cos_mat  = norms @ norms.T                  # (N, N)
    np.fill_diagonal(cos_mat, -np.inf)

    # Euclidean distance
    diff     = vecs[:, None, :] - vecs[None, :, :]
    euc_mat  = np.sqrt((diff ** 2).sum(-1))     # (N, N)
    np.fill_diagonal(euc_mat, np.inf)

    def _label(pid):
        if id_to_name and pid in id_to_name:
            return id_to_name[pid]
        return str(pid)

    # --- cosine top-N (highest sim, upper triangle only) ---
    upper_cos = np.triu(cos_mat, k=1).copy()
    flat      = upper_cos.ravel()
    top_idx   = np.argpartition(flat, -n)[-n:]
    top_idx   = top_idx[np.argsort(flat[top_idx])[::-1]]
    N         = len(ids)
    cosine_top = []
    for idx in top_idx:
        i, j = divmod(int(idx), N)
        if cos_mat[i, j] == -np.inf:
            continue
        cosine_top.append({
            "id_a": ids[i], "name_a": _label(ids[i]),
            "id_b": ids[j], "name_b": _label(ids[j]),
            "cosine_sim": float(cos_mat[i, j]),
        })

    # --- euclidean top-N (smallest dist, upper triangle only) ---
    upper_euc = np.triu(euc_mat, k=1).copy()
    upper_euc[upper_euc == np.inf] = np.finfo(np.float32).max
    flat_e    = upper_euc.ravel()
    bot_idx   = np.argpartition(flat_e, n)[:n]
    bot_idx   = bot_idx[np.argsort(flat_e[bot_idx])]
    eucl_top  = []
    for idx in bot_idx:
        i, j = divmod(int(idx), N)
        if euc_mat[i, j] == np.inf:
            continue
        eucl_top.append({
            "id_a": ids[i], "name_a": _label(ids[i]),
            "id_b": ids[j], "name_b": _label(ids[j]),
            "euclidean_dist": float(euc_mat[i, j]),
        })

    return cosine_top, eucl_top


def _print_similar_pairs(cosine_top, eucl_top, label):
    print(f"\n  Top 10 most similar players ({label}):")
    print(f"  {'─'*55}")
    print(f"  {'Cosine similarity':}")
    for k, p in enumerate(cosine_top, 1):
        print(f"    {k:2d}. {p['name_a']} ↔ {p['name_b']}  sim={p['cosine_sim']:.4f}")
    print(f"  {'Euclidean proximity':}")
    for k, p in enumerate(eucl_top, 1):
        print(f"    {k:2d}. {p['name_a']} ↔ {p['name_b']}  dist={p['euclidean_dist']:.4f}")


# =============================================================================
# HELPER: MISCLASSIFICATION TOP-10
# =============================================================================

def _misclassification_top10(y_true, y_pred):
    """
    Given arrays of true and predicted labels, return the top-10 (actual,
    predicted) error pairs sorted by frequency, excluding correct predictions.
    Returns a list of dicts: {actual, predicted, count}.
    """
    counts = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t != p:
            counts[(t, p)] += 1
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:10]
    return [{"actual": k[0], "predicted": k[1], "count": v} for k, v in top]


# =============================================================================
# Q1: TEAM EMBEDDINGS
# =============================================================================
team_results = []

if TEAM_KEY is not None:
    _banner("Q1 — Team embedding: sex, league")

    # Sex (binary) ─────────────────────────────────────────────────────────────
    r = probe_attribute(
        team_embedding_dict, team_to_sex,
        attr_name="sex", embed_name="team",
        label_points=LABEL_UMAP,
        id_to_name=team_id_to_name,
    )
    team_results.append(r)

    # League ───────────────────────────────────────────────────────────────────
    r = probe_attribute(
        team_embedding_dict, team_to_league,
        attr_name="league", embed_name="team",
        label_points=LABEL_UMAP,
        id_to_name=team_id_to_name,
    )
    team_results.append(r)

    # League within each sex (to test hierarchical structure) ──────────────────
    for sex_label in ("men", "women"):
        sex_ids = [tid for tid, sx in team_to_sex.items() if sx == sex_label]
        sub_embeds = {tid: team_embedding_dict[tid] for tid in sex_ids if tid in team_embedding_dict}
        sub_league = {tid: team_to_league[tid] for tid in sex_ids if tid in team_to_league}
        r = probe_attribute(
            sub_embeds, sub_league,
            attr_name=f"league_within_{sex_label}",
            embed_name="team",
            label_points=LABEL_UMAP,
            id_to_name=team_id_to_name,
        )
        team_results.append(r)
else:
    print("[SKIP] Team embedding absent (TEAM_KEY=None).")

# =============================================================================
# Q2: TEAM-SEASON EMBEDDINGS
# =============================================================================
ts_results = []

if TEAM_SEASON_KEY is not None:
    _banner("Q2 — Team-season embedding: team, league, season")

    # Build flat dicts of team-season embeddings using all available trajectories
    # We need all trajectories, not just one league.
    from tools import compute_trajectories
    if team_embedding_dict:
        all_trajectories = compute_trajectories(
            team_season_dict,
            team_season_weight,
            team_embedding_dict,
        )
    else:
        # No static team embeddings available; build ts_embeds directly
        all_trajectories = {
            team_name: {
                "seasons":   info["seasons"],
                "ts_embeddings": np.stack(
                    [team_season_weight[idx].numpy() if hasattr(team_season_weight[idx], "numpy")
                     else np.array(team_season_weight[idx])
                     for idx in info["season_indices"]], axis=0
                ).astype(np.float32),
                "combined_embeddings": None,
                "ts_displacements": [],
                "combined_displacements": [],
                "n_seasons": len(info["seasons"]),
            }
            for team_name, info in team_season_dict.items()
        }

    ts_embeds_dict, ts_team_attr, ts_season_attr = build_ts_embeds(
        all_trajectories, team_season_weight
    )

    # Build league attr for each team-season key
    # team_season_dict[team_name]["team_id"] is the StatsBomb id, same key space as team_to_league
    team_name_to_league = {}
    for team_name, info in team_season_dict.items():
        tid = info["team_id"]
        lg = team_to_league.get(tid)
        if lg:
            team_name_to_league[team_name] = lg

    ts_league_attr = {
        key: team_name_to_league.get(ts_team_attr[key])
        for key in ts_embeds_dict
        if ts_team_attr.get(key) in team_name_to_league
    }

    # Probe: season (all teams, cross-league) ─────────────────────────────────
    r = probe_attribute(
        ts_embeds_dict, ts_season_attr,
        attr_name="season", embed_name="team_season",
        also_euclidean_bc=True,
        label_points=LABEL_UMAP,
    )
    ts_results.append(r)

    # Probe: league ────────────────────────────────────────────────────────────
    r = probe_attribute(
        ts_embeds_dict, ts_league_attr,
        attr_name="league", embed_name="team_season",
        also_euclidean_bc=True,
        label_points=LABEL_UMAP,
    )
    ts_results.append(r)

    # Probe: team (should NOT encode team within a league) ─────────────────────
    # For tractability, limit to a single large league
    target_league = "England - Premier League"
    pl_keys = [k for k in ts_embeds_dict if ts_league_attr.get(k) == target_league]
    if pl_keys:
        pl_embeds  = {k: ts_embeds_dict[k] for k in pl_keys}
        pl_team    = {k: ts_team_attr[k]   for k in pl_keys}
        pl_season  = {k: ts_season_attr[k] for k in pl_keys}

        r = probe_attribute(
            pl_embeds, pl_team,
            attr_name=f"team_within_{target_league.replace(' ','_')}",
            embed_name="team_season",
            also_euclidean_bc=True,
            label_points=LABEL_UMAP,
        )
        ts_results.append(r)

        r = probe_attribute(
            pl_embeds, pl_season,
            attr_name=f"season_within_{target_league.replace(' ','_')}",
            embed_name="team_season",
            also_euclidean_bc=True,
            label_points=LABEL_UMAP,
        )
        ts_results.append(r)

    # Side-by-side heatmap: sorted by team vs sorted by season ─────────────────
    # Use one league for clarity
    if pl_keys:
        from tools import plot_team_vs_season_similarity
        fig = plot_team_vs_season_similarity(
            pl_embeds, pl_team, pl_season,
            title_prefix=f"Team-season | {target_league}",
            figsize=(18, 7),
        )
        save_fig(fig, "team_vs_season_heatmap")
else:
    print("[SKIP] Team-season embedding absent (TEAM_SEASON_KEY=None).")

# =============================================================================
# Q3: COMBINED (team + team-season) EMBEDDINGS
# =============================================================================
combined_results = []

if TEAM_KEY is not None and TEAM_SEASON_KEY is not None:
    _banner("Q3 — Combined embedding (team + team-season): league, season")

    # build_ts_embeds only gives ts embeddings; build combined flat dict manually
    combined_embeds_dict = {}
    comb_team_attr   = {}
    comb_season_attr = {}
    comb_league_attr = {}

    for team_name, info in all_trajectories.items():
        if info.get("combined_embeddings") is None:
            continue
        for i, season in enumerate(info["seasons"]):
            key = f"{team_name}_{season}"
            combined_embeds_dict[key] = info["combined_embeddings"][i]
            comb_team_attr[key]       = team_name
            comb_season_attr[key]     = season
            lg = team_name_to_league.get(team_name)
            if lg:
                comb_league_attr[key] = lg

    # Probe: season ────────────────────────────────────────────────────────────
    r = probe_attribute(
        combined_embeds_dict, comb_season_attr,
        attr_name="season", embed_name="combined",
        label_points=LABEL_UMAP,
    )
    combined_results.append(r)

    # Probe: league ────────────────────────────────────────────────────────────
    r = probe_attribute(
        combined_embeds_dict, comb_league_attr,
        attr_name="league", embed_name="combined",
        label_points=LABEL_UMAP,
    )
    combined_results.append(r)

    # Within-league season probe
    for lg_name in [target_league]:
        lg_keys = [k for k in combined_embeds_dict if comb_league_attr.get(k) == lg_name]
        if not lg_keys:
            continue
        lg_embeds = {k: combined_embeds_dict[k] for k in lg_keys}
        lg_season = {k: comb_season_attr[k]     for k in lg_keys}
        lg_team   = {k: comb_team_attr[k]        for k in lg_keys}

        r = probe_attribute(
            lg_embeds, lg_season,
            attr_name=f"season_within_{lg_name.replace(' ','_')}",
            embed_name="combined",
            label_points=LABEL_UMAP,
        )
        combined_results.append(r)

        r = probe_attribute(
            lg_embeds, lg_team,
            attr_name=f"team_within_{lg_name.replace(' ','_')}",
            embed_name="combined",
            label_points=LABEL_UMAP,
        )
        combined_results.append(r)
else:
    print("[SKIP] Combined embedding requires both TEAM_KEY and TEAM_SEASON_KEY.")

# =============================================================================
# Q4: PLAYER EMBEDDINGS
# =============================================================================
player_results = []

if PLAYER_KEY is not None:
    _banner("Q4 — Player embedding: sex, league, position group, position")

    # Sex ─────────────────────────────────────────────────────────────────────
    r = probe_attribute(
        player_sample_embeds, player_sample_sex,
        attr_name="sex", embed_name="player",
    )
    player_results.append(r)

    # League ──────────────────────────────────────────────────────────────────
    r = probe_attribute(
        player_sample_embeds, player_sample_league,
        attr_name="league", embed_name="player",
    )
    player_results.append(r)

    # Position group ───────────────────────────────────────────────────────────
    r = probe_attribute(
        player_sample_embeds, player_sample_pos_group,
        attr_name="position_group", embed_name="player",
    )
    player_results.append(r)

    # Team (within each sex, to keep n_classes tractable) ─────────────────────
    # PCA skipped: too many classes for a meaningful component-level test
    for sex_label in ("men", "women"):
        sex_pids = [p for p in sample if player_sample_sex.get(p) == sex_label]
        sub_embeds = {p: player_sample_embeds[p] for p in sex_pids}
        sub_team   = {p: player_sample_team[p]   for p in sex_pids}

        r = probe_attribute(
            sub_embeds, sub_team,
            attr_name=f"team_within_{sex_label}",
            embed_name="player",
            max_classes_for_heatmap=0,
            skip_pca=True,
        )
        player_results.append(r)

    # Position group within each sex ──────────────────────────────────────────
    for sex_label in ("men", "women"):
        sex_pids = [p for p in sample if player_sample_sex.get(p) == sex_label]
        sub_embeds  = {p: player_sample_embeds[p]    for p in sex_pids}
        sub_pos_grp = {p: player_sample_pos_group[p] for p in sex_pids}

        r = probe_attribute(
            sub_embeds, sub_pos_grp,
            attr_name=f"position_group_within_{sex_label}",
            embed_name="player",
            max_classes_for_heatmap=0,
        )
        player_results.append(r)

    # =========================================================================
    # Q4b — Per-league: position group LR + PLS + top similar pairs
    # =========================================================================
    _banner("Q4b — Player embedding: position group per league")

    # We use the full (unsampled) player set for per-league analysis so that
    # each league has as many players as possible.
    per_league_results = {}

    all_leagues_in_sample = sorted(set(player_sample_league.values()))

    for league_name in all_leagues_in_sample:
        lg_pids = [p for p in sample if player_sample_league.get(p) == league_name]
        if len(lg_pids) < MIN_SAMPLES_CLASS * 2:
            print(f"  [SKIP] {league_name}: only {len(lg_pids)} players.")
            continue

        lg_embeds  = {p: player_sample_embeds[p]    for p in lg_pids}
        lg_pos_grp = {p: player_sample_pos_group[p] for p in lg_pids}

        _subhead(f"League: {league_name}  (n={len(lg_pids)})")

        # --- LR ---
        lr_res = None
        try:
            lr_res = league_probe(
                lg_embeds, lg_pos_grp,
                n_folds=N_FOLDS, random_state=RANDOM_SEED,
            )
            acc    = lr_res["overall_accuracy"]
            chance = 1.0 / len(set(lg_pos_grp.values()))
            lift   = acc / chance if chance > 0 else float("nan")
            print(f"  LR accuracy: {acc:.3f}  (chance {chance:.3f}, lift {lift:.1f}×)")

            # Top-10 misclassifications
            miscls = _misclassification_top10(lr_res["y_true"], lr_res["y_pred_cv"])
            print(f"  Top misclassifications:")
            for m in miscls:
                print(f"    actual={m['actual']:20s}  predicted={m['predicted']:20s}  n={m['count']}")
        except Exception as e:
            print(f"  [WARN] LR failed for {league_name}: {e}")

        # --- PLS ---
        pls_res = None
        try:
            pls_res = league_pls_probe(
                lg_embeds, lg_pos_grp,
                random_state=RANDOM_SEED,
            )
            n_sig  = len(pls_res["significant_components"])
            n_comp = pls_res["n_components_used"]
            print(f"  PLS: {n_sig}/{n_comp} components significant (BH)")

            fig = _plot_component_significance(
                pls_res["results_table"],
                title=f"PLS: position group | {league_name}",
                x_col=None,
                x_label="Component index",
            )
            safe_lg = league_name.replace(" ", "_").replace("/", "-")
            save_fig(fig, f"pls_player_posgrp_{safe_lg}")
        except Exception as e:
            print(f"  [WARN] PLS failed for {league_name}: {e}")

        # --- Top similar pairs ---
        try:
            cos_top, euc_top = top_similar_pairs(lg_embeds, n=10)
            _print_similar_pairs(cos_top, euc_top, league_name)
        except Exception as e:
            print(f"  [WARN] Top pairs failed for {league_name}: {e}")
            cos_top, euc_top = [], []

        per_league_results[league_name] = {
            "n": len(lg_pids),
            "lr_accuracy":   lr_res["overall_accuracy"]            if lr_res  else None,
            "lr_chance":     1.0 / len(set(lg_pos_grp.values())),
            "lr_misclass":   _misclassification_top10(
                                 lr_res["y_true"], lr_res["y_pred_cv"]
                             )                                      if lr_res  else [],
            "lr_report":     lr_res["classification_report"]       if lr_res  else None,
            "pls_n_sig":     len(pls_res["significant_components"]) if pls_res else None,
            "pls_n_comp":    pls_res["n_components_used"]           if pls_res else None,
            "cosine_top10":  cos_top,
            "eucl_top10":    euc_top,
        }

    # =========================================================================
    # Q4c — Overall player top similar pairs (across full sample)
    # =========================================================================
    _banner("Q4c — Overall player top similar pairs")
    try:
        cos_top_all, euc_top_all = top_similar_pairs(player_sample_embeds, n=10)
        _print_similar_pairs(cos_top_all, euc_top_all, "all leagues")
    except Exception as e:
        print(f"  [WARN] Overall top pairs failed: {e}")
        cos_top_all, euc_top_all = [], []

else:
    print("[SKIP] Player embedding absent (PLAYER_KEY=None).")
    per_league_results = {}
    cos_top_all, euc_top_all = [], []

# =============================================================================
# SUMMARY FIGURE: LR accuracy for all probes
# =============================================================================
all_results = team_results + ts_results + combined_results + player_results

fig = plot_lr_summary(all_results, title="LR Accuracy Summary — All Probes")
if fig is not None:
    save_fig(fig, "lr_accuracy_summary")

# =============================================================================
# TEXT REPORT
# =============================================================================
_banner("Summary report")

lines = []
lines.append("EMBEDDING AUDIT — SUMMARY")
lines.append("=" * 70)
lines.append(f"Checkpoint:  {CHECKPOINT_PATH}")
lines.append(f"Team key:    {TEAM_KEY}")
lines.append(f"Season key:  {TEAM_SEASON_KEY}")
lines.append(f"Player key:  {PLAYER_KEY}")
lines.append("")

section_map = [
    ("Q1 — Team embedding",           team_results),
    ("Q2 — Team-season embedding",     ts_results),
    ("Q3 — Combined embedding",        combined_results),
    ("Q4 — Player embedding",          player_results),
]

for section_title, results in section_map:
    lines.append(f"\n{'─'*70}")
    lines.append(f"  {section_title}")
    lines.append(f"{'─'*70}")
    for r in results:
        if r is None:
            continue
        tag    = f"{r['embed_name']} → {r['attr_name']}"
        acc    = r.get("lr_accuracy")
        chance = r["chance"]
        bc     = r.get("block_coherence")
        n_pca  = r.get("pca_n_significant")
        n_pls  = r.get("pls_n_significant")
        n_comp = r.get("pca_n_components")
        umap_r = r.get("umap_run", False)

        lines.append(f"\n  {tag}")
        lines.append(f"    n={r['n_items']}, classes={r['n_classes']}, "
                     f"chance={chance:.2%}")
        if acc is not None:
            lift = acc / chance if chance > 0 else float("nan")
            lines.append(f"    LR accuracy: {acc:.3f}  (lift {lift:.1f}×)")
        if bc is not None:
            lines.append(f"    Block coherence (cosine): {bc:+.4f}")
        bc_euc = r.get("block_coherence_euclidean")
        if bc_euc is not None:
            lines.append(f"    Block coherence (euclidean): {bc_euc:+.4f}")
        if n_pca is not None:
            lines.append(f"    PCA: {n_pca}/{n_comp} components significant (BH)")
        if n_pls is not None:
            n_pls_comp = r.get("pls_n_components","?")
            lines.append(f"    PLS: {n_pls}/{n_pls_comp} components significant (BH)")
        if umap_r:
            lines.append(f"    UMAP: run (LR > {UMAP_ACCURACY_GATE:.0%} gate)")

        # Interpretation hint
        if acc is not None and bc is not None:
            strong = acc >= 0.7 and bc > 0.05
            moderate = 0.4 <= acc < 0.7 or (0.01 < bc <= 0.05)
            if strong:
                lines.append(f"    ✔  Strong encoding signal")
            elif moderate:
                lines.append(f"    ~  Moderate encoding signal")
            else:
                lines.append(f"    ✗  Weak/absent encoding signal")

# Per-league position group results
if per_league_results:
    lines.append(f"\n\n{'='*70}")
    lines.append("Q4b — POSITION GROUP PER LEAGUE")
    lines.append("=" * 70)
    for lg_name, lg_r in per_league_results.items():
        lines.append(f"\n  {lg_name}  (n={lg_r['n']})")
        acc = lg_r["lr_accuracy"]
        if acc is not None:
            chance = lg_r["lr_chance"]
            lift   = acc / chance if chance > 0 else float("nan")
            lines.append(f"    LR accuracy: {acc:.3f}  (chance {chance:.3f}, lift {lift:.1f}×)")
        pls_sig  = lg_r["pls_n_sig"]
        pls_comp = lg_r["pls_n_comp"]
        if pls_sig is not None:
            lines.append(f"    PLS: {pls_sig}/{pls_comp} components significant (BH)")
        if lg_r["lr_misclass"]:
            lines.append(f"    Top misclassifications:")
            for m in lg_r["lr_misclass"]:
                lines.append(f"      actual={m['actual']:20s}  predicted={m['predicted']:20s}  n={m['count']}")
        if lg_r["cosine_top10"]:
            lines.append(f"    Top 10 pairs (cosine):")
            for p in lg_r["cosine_top10"]:
                lines.append(f"      {p['name_a']} ↔ {p['name_b']}  sim={p['cosine_sim']:.4f}")
        if lg_r["eucl_top10"]:
            lines.append(f"    Top 10 pairs (euclidean):")
            for p in lg_r["eucl_top10"]:
                lines.append(f"      {p['name_a']} ↔ {p['name_b']}  dist={p['euclidean_dist']:.4f}")

# Overall top similar pairs
if cos_top_all or euc_top_all:
    lines.append(f"\n\n{'='*70}")
    lines.append("Q4c — OVERALL PLAYER TOP SIMILAR PAIRS")
    lines.append("=" * 70)
    lines.append("\n  Top 10 (cosine similarity):")
    for p in cos_top_all:
        lines.append(f"    {p['name_a']} ↔ {p['name_b']}  sim={p['cosine_sim']:.4f}")
    lines.append("\n  Top 10 (euclidean proximity):")
    for p in euc_top_all:
        lines.append(f"    {p['name_a']} ↔ {p['name_b']}  dist={p['euclidean_dist']:.4f}")

# Classification reports (verbose)
lines.append(f"\n\n{'='*70}")
lines.append("DETAILED CLASSIFICATION REPORTS")
lines.append("=" * 70)
for r in all_results:
    if r is None or r.get("lr_report") is None:
        continue
    lines.append(f"\n[{r['embed_name']} → {r['attr_name']}]")
    lines.append(r["lr_report"])
if per_league_results:
    lines.append(f"\n[Per-league position group reports]")
    for lg_name, lg_r in per_league_results.items():
        if lg_r.get("lr_report"):
            lines.append(f"\n  --- {lg_name} ---")
            lines.append(lg_r["lr_report"])

report_text = "\n".join(lines)
print(report_text)

report_path = os.path.join(OUTPUT_DIR, "audit_summary.txt")
with open(report_path, "w") as f:
    f.write(report_text)

print(f"\nReport saved to: {report_path}")
print(f"Plots saved to:  {OUTPUT_DIR}/")
print(f"Total plots:     {_plot_counter[0]}")