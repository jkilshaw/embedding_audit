import pickle
from pathlib import Path
import numpy as np
import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Hashable, List, Optional, Tuple
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.colors import to_rgba
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import umap
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.multitest import multipletests
from sklearn.cross_decomposition import PLSRegression
import random
from collections import defaultdict


def load_pickle(path):
    with open(path,'rb') as f:
        return pickle.load(f)

def make_league_dict(teams, matches):
    """
    Given a list of teams, and a list of match metadata dictionaries, make a direct dictionary of teams to leagues.
    teams: list, team ids as integers
    Each match dict will have keys:
        league
        home_team_id
        away_team_id
    Using these keys, build a dictionary of team_id to league name
    """
    team_set = set(teams)
    team_to_league = {}

    for match in matches:
        league = match["league"]
        for team_id in (match["home_team_id"], match["away_team_id"]):
            if team_id in team_set and team_id not in team_to_league:
                team_to_league[team_id] = league

    missing = team_set - team_to_league.keys()
    if missing:
        print(f"[WARN] {len(missing)} team(s) not found in any match: {missing}")

    return team_to_league

def cos_sim_mat(
    embeds_dict: Dict[Hashable, np.ndarray],
    attribute_dict: Optional[Dict[Hashable, str]] = None,
) -> Tuple[np.ndarray, List[Hashable], Optional[List[str]]]:
    """
    Compute a pairwise cosine similarity matrix, optionally sorted into
    attribute blocks.

    Args:
        embeds_dict:    dict mapping id -> embedding vector
        attribute_dict: optional dict mapping id -> discrete attribute label
                        (e.g. league, sex). If provided, rows/cols are sorted
                        so that ids sharing an attribute are contiguous.

    Returns:
        sim_matrix: (N, N) cosine similarity matrix
        ids:        ordered list of ids corresponding to rows/cols
        attributes: ordered list of attribute labels (None if no attribute_dict)
    """
    ids = list(embeds_dict.keys())

    if attribute_dict is not None:
        # Sort by attribute so same-attribute ids are contiguous blocks
        ids.sort(key=lambda x: (attribute_dict.get(x, ""), x))
        attributes = [attribute_dict.get(i) for i in ids]
    else:
        attributes = None

    vecs = np.stack([embeds_dict[i] for i in ids], axis=0)  # (N, D)
    vecs = normalize(vecs, norm="l2", axis=1)
    sim_matrix = vecs @ vecs.T  # (N, N)

    return sim_matrix, ids, attributes


def pca(
    embeds_dict: Dict[Hashable, np.ndarray],
    n_components: Optional[int] = None,
) -> Tuple[np.ndarray, List[Hashable], PCA]:
    """
    Perform PCA on a set of embeddings.

    Args:
        embeds_dict:  dict mapping id -> embedding vector
        n_components: number of components to keep (None = keep all)

    Returns:
        coords:     (N, n_components) projected coordinates
        ids:        ordered list of ids corresponding to rows
        pca_model:  fitted sklearn PCA object (for explained variance etc.)
    """
    ids = list(embeds_dict.keys())
    vecs = np.stack([embeds_dict[i] for i in ids], axis=0)  # (N, D)

    pca_model = PCA(n_components=n_components)
    coords = pca_model.fit_transform(vecs)  # (N, n_components)

    return coords, ids, pca_model


def umap_embed(
    embeds_dict: Dict[Hashable, np.ndarray],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 1337,
) -> Tuple[np.ndarray, List[Hashable], umap.UMAP]:
    """
    Perform UMAP dimensionality reduction on a set of embeddings.

    Args:
        embeds_dict:  dict mapping id -> embedding vector
        n_components: number of dimensions to reduce to (default 2)
        n_neighbors:  UMAP neighbourhood size — larger = more global structure
        min_dist:     minimum distance between points in low-dim space
        metric:       distance metric (cosine works well for embeddings)
        random_state: random seed for reproducibility

    Returns:
        coords:     (N, n_components) projected coordinates
        ids:        ordered list of ids corresponding to rows
        umap_model: fitted UMAP object
    """
    ids = list(embeds_dict.keys())
    vecs = np.stack([embeds_dict[i] for i in ids], axis=0)  # (N, D)

    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    coords = umap_model.fit_transform(vecs)  # (N, n_components)

    return coords, ids, umap_model



# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------

def _make_attribute_colormap(
    attributes: List[str],
) -> Tuple[Dict[str, tuple], List[str]]:
    """Map unique attribute labels to colours from the tab20 palette."""
    unique = sorted(set(a for a in attributes if a is not None))
    cmap = plt.cm.get_cmap("tab20", len(unique))
    colour_map = {label: cmap(i) for i, label in enumerate(unique)}
    return colour_map, unique


# -----------------------------------------------------------------------------
# Heatmap
# -----------------------------------------------------------------------------

def plot_heatmap(
    sim_matrix: np.ndarray,
    ids: List[Hashable],
    attributes: Optional[List[str]] = None,
    title: str = "Cosine Similarity Matrix",
    figsize: Tuple[int, int] = (12, 10),
) -> Figure:
    """
    Plot a cosine similarity heatmap with optional attribute block delineation.

    Args:
        sim_matrix:  (N, N) cosine similarity matrix
        ids:         ordered list of ids corresponding to rows/cols
        attributes:  optional ordered list of attribute labels per id.
                     When provided, block boundaries are drawn and tick
                     labels are coloured by attribute.
        title:       plot title
        figsize:     figure size in inches

    Returns:
        fig: matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    ax.set_xticklabels([str(i) for i in ids], rotation=90, fontsize=6)
    ax.set_yticklabels([str(i) for i in ids], fontsize=6)

    if attributes is not None:
        colour_map, unique_attrs = _make_attribute_colormap(attributes)

        # Colour tick labels
        for tick, attr in zip(ax.get_xticklabels(), attributes):
            tick.set_color(colour_map.get(attr, (0, 0, 0, 1)))
        for tick, attr in zip(ax.get_yticklabels(), attributes):
            tick.set_color(colour_map.get(attr, (0, 0, 0, 1)))

        # Draw block boundary lines wherever the attribute changes
        boundaries = [
            i + 0.5
            for i in range(len(attributes) - 1)
            if attributes[i] != attributes[i + 1]
        ]
        for b in boundaries:
            ax.axhline(b, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(b, color="black", linewidth=0.8, linestyle="--")

        # Legend
        handles = [
            mpatches.Patch(color=colour_map[a], label=a) for a in unique_attrs
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.15, 1), loc="upper left",
                  fontsize=7, framealpha=0.8)

    ax.set_title(title)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# PCA
# -----------------------------------------------------------------------------

def plot_pca(
    coords: np.ndarray,
    ids: List[Hashable],
    pca_model,
    components: Tuple[int, int] = (0, 1),
    attribute_dict: Optional[Dict[Hashable, str]] = None,
    title: str = "PCA",
    figsize: Tuple[int, int] = (9, 7),
    label_points: bool = False,
) -> Figure:
    """
    Scatter plot of two PCA components, optionally coloured by attribute.

    Args:
        coords:         (N, n_components) array from pca()
        ids:            ordered list of ids corresponding to rows
        pca_model:      fitted sklearn PCA object (for explained variance)
        components:     tuple of two 0-based component indices to plot
        attribute_dict: optional dict mapping id -> discrete attribute label
        title:          plot title
        figsize:        figure size in inches
        label_points:   annotate each point with its id

    Returns:
        fig: matplotlib Figure object
    """
    cx, cy = components
    xs = coords[:, cx]
    ys = coords[:, cy]

    fig, ax = plt.subplots(figsize=figsize)

    if attribute_dict is not None:
        attributes = [attribute_dict.get(i) for i in ids]
        colour_map, unique_attrs = _make_attribute_colormap(attributes)
        colours = [colour_map.get(a, (0.5, 0.5, 0.5, 1.0)) for a in attributes]
        scatter = ax.scatter(xs, ys, c=colours, s=40, alpha=0.8, linewidths=0)
        handles = [
            mpatches.Patch(color=colour_map[a], label=a) for a in unique_attrs
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=8, framealpha=0.8)
    else:
        ax.scatter(xs, ys, s=40, alpha=0.8, linewidths=0)

    if label_points:
        for x, y, id_ in zip(xs, ys, ids):
            ax.annotate(str(id_), (x, y), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ev = pca_model.explained_variance_ratio_
    ax.set_xlabel(f"PC{cx + 1} ({ev[cx]:.1%} var)", fontsize=10)
    ax.set_ylabel(f"PC{cy + 1} ({ev[cy]:.1%} var)", fontsize=10)
    ax.set_title(title)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# UMAP
# -----------------------------------------------------------------------------

def plot_umap(
    coords: np.ndarray,
    ids: List[Hashable],
    components: Tuple[int, int] = (0, 1),
    attribute_dict: Optional[Dict[Hashable, str]] = None,
    title: str = "UMAP",
    figsize: Tuple[int, int] = (9, 7),
    label_points: bool = False,
) -> Figure:
    """
    Scatter plot of two UMAP dimensions, optionally coloured by attribute.

    Unlike PCA, UMAP axes have no intrinsic meaning so no axis labels are
    shown beyond the dimension index. Relative distances within the plot are
    meaningful; distances between clusters are not.

    Args:
        coords:         (N, n_components) array from umap_embed()
        ids:            ordered list of ids corresponding to rows
        components:     tuple of two 0-based dimension indices to plot
        attribute_dict: optional dict mapping id -> discrete attribute label.
                        Missing keys are rendered in grey and excluded from
                        the legend, allowing sparse highlighting of a subset
                        of points.
        title:          plot title
        figsize:        figure size in inches
        label_points:   annotate each point with its id

    Returns:
        fig: matplotlib Figure object
    """
    cx, cy = components
    xs = coords[:, cx]
    ys = coords[:, cy]

    fig, ax = plt.subplots(figsize=figsize)

    if attribute_dict is not None:
        attributes = [attribute_dict.get(i) for i in ids]  # None for missing keys

        # Build colormap only for ids that are present in attribute_dict
        present_attrs = [a for a in attributes if a is not None]
        colour_map, unique_attrs = _make_attribute_colormap(present_attrs)

        # Split into grey (missing) and coloured (present) — plot grey first
        # so coloured points render on top
        grey   = [(x, y, id_) for x, y, id_, a in zip(xs, ys, ids, attributes) if a is None]
        coloured = [(x, y, id_, a) for x, y, id_, a in zip(xs, ys, ids, attributes) if a is not None]

        if grey:
            gx, gy, _ = zip(*grey)
            ax.scatter(gx, gy, c=["#cccccc"], s=20, alpha=0.4, linewidths=0,
                       zorder=1, label="_nolegend_")

        if coloured:
            cx_, cy_, cids, cattrs = zip(*coloured)
            colours = [colour_map[a] for a in cattrs]
            ax.scatter(cx_, cy_, c=colours, s=40, alpha=0.9, linewidths=0, zorder=2)

        handles = [
            mpatches.Patch(color=colour_map[a], label=a) for a in unique_attrs
        ]
        ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
                  fontsize=8, framealpha=0.8)
    else:
        ax.scatter(xs, ys, s=40, alpha=0.8, linewidths=0)

    if label_points:
        for x, y, id_ in zip(xs, ys, ids):
            ax.annotate(str(id_), (x, y), fontsize=5, alpha=0.7,
                        xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel(f"UMAP dim {cx + 1}", fontsize=10)
    ax.set_ylabel(f"UMAP dim {cy + 1}", fontsize=10)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig

from typing import Dict, Hashable, List, Optional, Tuple

import numpy as np


def top_n_similarities(
    sim_matrix: np.ndarray,
    ids: List[Hashable],
    n: int = 10,
    attributes: Optional[List[str]] = None,
    cross_attribute_only: bool = False,
) -> Tuple[List[Dict], Dict[Hashable, List[Dict]]]:
    """
    Find the top N cosine similarities both globally and per entity.

    Args:
        sim_matrix:           (N, N) cosine similarity matrix
        ids:                  ordered list of ids corresponding to rows/cols
        n:                    number of top pairs to return
        attributes:           optional ordered list of attribute labels per id
        cross_attribute_only: if True, only consider pairs from different
                              attributes (e.g. cross-league similarities)

    Returns:
        global_top:  list of dicts, length N, sorted descending by similarity
                     each dict has keys: id_a, id_b, similarity,
                     and optionally attribute_a, attribute_b
        per_entity_top: dict mapping each id to its top N most similar others,
                        same dict format as global_top (excluding self)
    """
    n_ids = len(ids)
    has_attrs = attributes is not None

    def _make_record(i, j):
        rec = {
            "id_a":       ids[i],
            "id_b":       ids[j],
            "similarity": float(sim_matrix[i, j]),
        }
        if has_attrs:
            rec["attribute_a"] = attributes[i]
            rec["attribute_b"] = attributes[j]
        return rec

    def _include_pair(i, j):
        if i == j:
            return False
        if cross_attribute_only and has_attrs and attributes[i] == attributes[j]:
            return False
        return True

    # --- Global top N ---
    # Zero out diagonal and lower triangle to avoid duplicate pairs
    upper = np.triu(sim_matrix, k=1).copy()
    if cross_attribute_only and has_attrs:
        for i in range(n_ids):
            for j in range(i + 1, n_ids):
                if attributes[i] == attributes[j]:
                    upper[i, j] = -np.inf

    flat = upper.ravel()
    top_flat_idx = np.argpartition(flat, -n)[-n:]
    top_flat_idx = top_flat_idx[np.argsort(flat[top_flat_idx])[::-1]]
    global_top = [
        _make_record(*divmod(idx, n_ids))
        for idx in top_flat_idx
        if flat[idx] > -np.inf
    ]

    # --- Per-entity top N ---
    per_entity_top = {}
    for i, id_ in enumerate(ids):
        row = sim_matrix[i].copy()
        row[i] = -np.inf  # exclude self
        if cross_attribute_only and has_attrs:
            for j in range(n_ids):
                if j != i and attributes[j] == attributes[i]:
                    row[j] = -np.inf

        top_idx = np.argpartition(row, -n)[-n:]
        top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
        per_entity_top[id_] = [
            _make_record(i, j)
            for j in top_idx
            if row[j] > -np.inf
        ]

    return global_top, per_entity_top

def league_probe(
    embeds_dict: Dict[Hashable, np.ndarray],
    team_to_league: Dict[Hashable, str],
    n_folds: int = 5,
    max_iter: int = 1000,
    random_state: int = 1337,
) -> Dict:
    """
    Fit a logistic regression probe on team embeddings to predict league.
    Evaluates via stratified k-fold cross-validation.

    Args:
        embeds_dict:    dict mapping team id -> embedding vector
        team_to_league: dict mapping team id -> league name
        n_folds:        number of CV folds
        max_iter:       max iterations for logistic regression solver
        random_state:   random seed

    Returns:
        dict with keys:
            model:              fitted LogisticRegression on all data
            label_encoder:      fitted LabelEncoder (int <-> league name)
            ids:                ordered list of team ids used
            y_true:             true league labels (strings)
            y_pred_cv:          cross-validated predictions (strings)
            overall_accuracy:   float
            per_league_accuracy: dict mapping league name -> accuracy
            confusion_matrix:   (n_classes, n_classes) np.ndarray,
                                rows=true, cols=predicted
            league_names:       ordered list of league names (confusion matrix axis)
            classification_report: full sklearn classification report string
            cv_fold_accuracies: per-fold accuracy scores
            coef:               (n_classes, n_embd) coefficient matrix from
                                full-data fit — magnitude indicates which
                                embedding dims drive each league prediction
    """
    # Align ids that appear in both dicts
    ids = [i for i in embeds_dict if i in team_to_league]
    if len(ids) == 0:
        raise ValueError("No overlap between embeds_dict and team_to_league keys.")

    X = np.stack([embeds_dict[i] for i in ids], axis=0).astype(np.float32)
    leagues = [team_to_league[i] for i in ids]

    le = LabelEncoder()
    y = le.fit_transform(leagues)

    # Scale features — logistic regression is sensitive to feature scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
    )

    # Cross-validated predictions
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)

    # Per-fold accuracies
    fold_accs = []
    for train_idx, val_idx in cv.split(X_scaled, y):
        clf_fold = LogisticRegression(
            max_iter=max_iter, random_state=random_state,
            multi_class="multinomial", solver="lbfgs", C=1.0,
        )
        clf_fold.fit(X_scaled[train_idx], y[train_idx])
        fold_accs.append(accuracy_score(y[val_idx], clf_fold.predict(X_scaled[val_idx])))

    # Fit on full data for model + coefficients
    clf.fit(X_scaled, y)

    # Metrics
    y_true_labels = le.inverse_transform(y)
    y_pred_labels = le.inverse_transform(y_pred_cv)

    overall_acc = accuracy_score(y, y_pred_cv)

    per_league_acc = {}
    for league in le.classes_:
        mask = y_true_labels == league
        if mask.sum() > 0:
            per_league_acc[league] = accuracy_score(
                y_true_labels[mask], y_pred_labels[mask]
            )

    cm = confusion_matrix(y, y_pred_cv, labels=list(range(len(le.classes_))))
    report = classification_report(y_true_labels, y_pred_labels)

    return {
        "model":                 clf,
        "scaler":                scaler,
        "label_encoder":         le,
        "ids":                   ids,
        "y_true":                y_true_labels,
        "y_pred_cv":             y_pred_labels,
        "overall_accuracy":      overall_acc,
        "per_league_accuracy":   per_league_acc,
        "confusion_matrix":      cm,
        "league_names":          list(le.classes_),
        "classification_report": report,
        "cv_fold_accuracies":    fold_accs,
        "coef":                  clf.coef_,  # (n_leagues, n_embd)
    }

def league_component_probe(
    embeds_dict: Dict[Hashable, np.ndarray],
    team_to_league: Dict[Hashable, str],
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = 0.95,
    alpha: float = 0.05,
    random_state: int = 1337,
) -> Dict:
    """
    Test which PCA components of team embeddings encode statistically
    significant information about league membership, using multinomial
    logistic regression (MNLogit) and Benjamini-Hochberg FDR correction.

    Component count is determined by either:
      - n_components: fixed number of PCA components (takes priority if set)
      - variance_threshold: keep enough components to explain this fraction
        of total variance (default 0.95)

    Args:
        embeds_dict:        dict mapping team id -> embedding vector
        team_to_league:     dict mapping team id -> league name
        n_components:       fixed number of PCA components (overrides threshold)
        variance_threshold: cumulative explained variance threshold (0-1)
        alpha:              FDR significance threshold
        random_state:       random seed for PCA

    Returns:
        dict with keys:
            pca:                    fitted PCA object
            pca_coords:             (N, n_components) projected coordinates
            n_components_used:      number of components passed to MNLogit
            explained_variance:     explained variance ratio per component
            cumulative_variance:    cumulative explained variance per component
            label_encoder:          fitted LabelEncoder
            ids:                    ordered list of team ids used

            results_table:          pd.DataFrame with one row per component:
                                      component, explained_var, p_value_raw,
                                      p_value_corrected, significant, max_coef_league
            significant_components: list of component indices (0-based) that
                                    survive BH correction
            loadings:               (n_components, n_embd) PCA loading matrix —
                                    use this to interpret significant components
                                    in the original embedding space
            mnlogit_result:         fitted statsmodels MNLogit result object
                                    for full inspection if needed
    """
    # --- Align ids ---
    ids = [i for i in embeds_dict if i in team_to_league]
    if len(ids) == 0:
        raise ValueError("No overlap between embeds_dict and team_to_league keys.")

    X = np.stack([embeds_dict[i] for i in ids], axis=0).astype(np.float64)
    leagues = [team_to_league[i] for i in ids]

    le = LabelEncoder()
    y = le.fit_transform(leagues)

    # --- Scale ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- PCA: determine n_components ---
    max_components = min(len(ids), X_scaled.shape[1])
    pca_full = PCA(n_components=max_components, random_state=random_state)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is not None:
        n = min(n_components, max_components)
    else:
        n = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n = min(n, max_components)

    print(f"Using {n} PCA components "
          f"(explains {cumvar[n - 1]:.1%} of variance)")

    pca = PCA(n_components=n, random_state=random_state)
    coords = pca.fit_transform(X_scaled)  # (N, n)

    # --- MNLogit ---
    # statsmodels needs float64 and an intercept column
    X_mnlogit = np.hstack([np.ones((len(ids), 1)), coords])

    model = MNLogit(y, X_mnlogit)
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)

    # --- Extract p-values per component ---
    # result.pvalues shape: (n_params, n_classes-1)
    # Row 0 is the intercept; rows 1..n are the components.
    # We take the max p-value across classes for each component as a
    # conservative per-component summary (if any class shows significance,
    # the component is informative).
    pvals_per_component = result.pvalues[1:, :]  # (n, n_classes-1)
    pvals_min = pvals_per_component.min(axis=1)  # most significant class

    # --- Benjamini-Hochberg correction ---
    reject, pvals_corrected, _, _ = multipletests(
        pvals_min, alpha=alpha, method="fdr_bh"
    )
    significant_components = [i for i, r in enumerate(reject) if r]

    # --- Which league drives each component most? ---
    # Use the coefficient with the largest absolute value per component
    coefs = result.params[1:, :]  # (n, n_classes-1)
    max_coef_class_idx = np.abs(coefs).max(axis=1).argmax() if coefs.shape[1] > 0 else None
    max_coef_leagues = []
    for i in range(n):
        class_idx = int(np.abs(coefs[i]).argmax())
        # MNLogit uses the first class as reference; class indices offset by 1
        league_idx = class_idx + 1 if class_idx + 1 < len(le.classes_) else class_idx
        max_coef_leagues.append(le.classes_[league_idx])

    # --- Results table ---
    results_table = pd.DataFrame({
        "component":          range(n),
        "explained_var":      pca.explained_variance_ratio_,
        "cumulative_var":     np.cumsum(pca.explained_variance_ratio_),
        "p_value_raw":        pvals_min,
        "p_value_corrected":  pvals_corrected,
        "significant":        reject,
        "max_coef_league":    max_coef_leagues,
    })

    return {
        "pca":                    pca,
        "pca_coords":             coords,
        "n_components_used":      n,
        "explained_variance":     pca.explained_variance_ratio_,
        "cumulative_variance":    np.cumsum(pca.explained_variance_ratio_),
        "label_encoder":          le,
        "ids":                    ids,
        "results_table":          results_table,
        "significant_components": significant_components,
        "loadings":               pca.components_,  # (n, n_embd)
        "mnlogit_result":         result,
    }

def league_pls_probe(
    embeds_dict: Dict[Hashable, np.ndarray],
    team_to_league: Dict[Hashable, str],
    n_components: Optional[int] = None,
    alpha: float = 0.05,
    random_state: int = 1337,
) -> Dict:
    """
    Test which PLS components of team embeddings encode statistically
    significant information about league membership.

    Unlike PCA, PLS finds components that maximise covariance with the
    league labels directly, so it is better suited to finding league signal
    that lives in low-variance directions of the embedding space.

    Y is one-hot encoded league membership. PLSRegression is used since
    sklearn does not have a dedicated PLS-DA, but PLS on a one-hot Y matrix
    is equivalent to PLS-DA.

    n_components defaults to min(n_teams - 1, n_leagues - 1) if not set,
    which is the maximum meaningful rank for this problem.

    Args:
        embeds_dict:    dict mapping team id -> embedding vector
        team_to_league: dict mapping team id -> league name
        n_components:   number of PLS components (see above for default)
        alpha:          FDR significance threshold for BH correction
        random_state:   unused (PLS is deterministic), kept for API consistency

    Returns:
        dict with keys:
            pls:                    fitted PLSRegression object
            pls_coords:             (N, n_components) X scores (team projections)
            n_components_used:      int
            label_encoder:          fitted LabelEncoder
            ids:                    ordered list of team ids used
            results_table:          pd.DataFrame with one row per component:
                                      component, p_value_raw, p_value_corrected,
                                      significant, max_coef_league
            significant_components: list of 0-based component indices
            x_loadings:             (n_embd, n_components) X loadings —
                                    use to interpret which embedding dims
                                    drive each component
            mnlogit_result:         fitted statsmodels MNLogit result object
    """
    # --- Align ids ---
    ids = [i for i in embeds_dict if i in team_to_league]
    if len(ids) == 0:
        raise ValueError("No overlap between embeds_dict and team_to_league keys.")

    X = np.stack([embeds_dict[i] for i in ids], axis=0).astype(np.float64)
    leagues = [team_to_league[i] for i in ids]

    le = LabelEncoder()
    y_int = le.fit_transform(leagues)
    n_leagues = len(le.classes_)

    # One-hot encode Y for PLS-DA
    ohe = OneHotEncoder(sparse_output=False)
    Y = ohe.fit_transform(y_int.reshape(-1, 1))  # (N, n_leagues)

    # --- Scale X ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Determine n_components ---
    max_components = min(len(ids) - 1, n_leagues - 1)
    if n_components is not None:
        n = min(n_components, max_components)
    else:
        n = max_components

    print(f"Using {n} PLS components (max meaningful = min(N-1, n_leagues-1) = {max_components})")

    # --- Fit PLS ---
    pls = PLSRegression(n_components=n, scale=False)
    pls.fit(X_scaled, Y)
    coords = pls.transform(X_scaled)  # (N, n) — X scores

    # --- MNLogit on PLS scores ---
    X_mnlogit = np.hstack([np.ones((len(ids), 1)), coords])
    model = MNLogit(y_int, X_mnlogit)
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)

    # --- Per-component p-values (min across league contrasts) ---
    pvals_per_component = result.pvalues[1:, :]  # (n, n_classes-1)
    pvals_min = pvals_per_component.min(axis=1)

    # --- BH correction ---
    reject, pvals_corrected, _, _ = multipletests(
        pvals_min, alpha=alpha, method="fdr_bh"
    )
    significant_components = [i for i, r in enumerate(reject) if r]

    # --- Which league drives each component most? ---
    coefs = result.params[1:, :]  # (n, n_classes-1)
    max_coef_leagues = []
    for i in range(n):
        class_idx = int(np.abs(coefs[i]).argmax())
        league_idx = min(class_idx + 1, len(le.classes_) - 1)
        max_coef_leagues.append(le.classes_[league_idx])

    # --- Results table ---
    results_table = pd.DataFrame({
        "component":         range(n),
        "p_value_raw":       pvals_min,
        "p_value_corrected": pvals_corrected,
        "significant":       reject,
        "max_coef_league":   max_coef_leagues,
    })

    return {
        "pls":                    pls,
        "pls_coords":             coords,
        "n_components_used":      n,
        "label_encoder":          le,
        "ids":                    ids,
        "results_table":          results_table,
        "significant_components": significant_components,
        "x_loadings":             pls.x_loadings_,  # (n_embd, n_components)
        "mnlogit_result":         result,
    }

def build_team_season_dict(team_season_mapping, embedding_matrix):
    """
    Build a dictionary mapping team name to its id, seasons, and the
    corresponding row indices into the team-season embedding matrix.

    Only includes seasons where the embedding is non-zero (i.e. the team
    actually has data for that season).

    Args:
        team_season_mapping: dict with a 'rows' key mapping row index -> row dict.
                             Each row dict has keys:
                                 team_season_id, season_id, season_token,
                                 team_vocab_id, team_sb_id, team_token
        embedding_matrix:    np.ndarray or torch.Tensor, shape (n_team_seasons, n_embd)
                             Used to filter out uninitialised (all-zero) rows.

    Returns:
        dict mapping team_name -> {
            "team_id":        int,           # StatsBomb integer team id
            "seasons":        [str, ...],    # season tokens, e.g. "2015_2016"
            "season_indices": [int, ...],    # corresponding row indices into embedding_matrix
        }
        Seasons are sorted chronologically by season_token.
    """
    if hasattr(embedding_matrix, "numpy"):
        emb = embedding_matrix.numpy()
    else:
        emb = np.array(embedding_matrix)

    result = {}

    for row_idx, row in enumerate(team_season_mapping["rows"]):
        # Skip special tokens
        if row["team_token"] in ("<UNK>", "game"):
            continue

        # Skip uninitialised embeddings
        if not np.any(emb[row_idx]):
            continue

        # Parse team name from token string e.g. "749:tottenham_hotspur_women"
        team_name = row["team_token"].split(":", 1)[1] if ":" in row["team_token"] else row["team_token"]
        team_id   = row["team_sb_id"]
        season    = row["season_token"]
        idx       = row["team_season_id"]

        if team_name not in result:
            result[team_name] = {
                "team_id":        team_id,
                "seasons":        [],
                "season_indices": [],
            }

        result[team_name]["seasons"].append(season)
        result[team_name]["season_indices"].append(idx)

    # Sort each team's seasons chronologically by season token
    for team_name, info in result.items():
        sorted_pairs = sorted(zip(info["seasons"], info["season_indices"]))
        info["seasons"]        = [s for s, _ in sorted_pairs]
        info["season_indices"] = [i for _, i in sorted_pairs]
        info["n_seasons"] = len(info["seasons"])

    return result

# -----------------------------------------------------------------------------
# Core displacement computation
# -----------------------------------------------------------------------------

def compute_trajectories(
    team_season_dict: Dict,
    embedding_matrix: np.ndarray,
    static_team_embeddings: Dict[str, np.ndarray],
    teams: Optional[List[str]] = None,
) -> Dict[str, Dict]:
    """
    Compute season-to-season cosine displacement trajectories for each team,
    in both the team-season-only and combined (team + team-season) spaces.

    Args:
        team_season_dict:       output of build_team_season_dict()
        embedding_matrix:       (n_team_seasons, n_embd) team-season embedding matrix
        static_team_embeddings: dict mapping team_name -> (n_embd,) static embedding
        teams:                  optional list of team names to restrict to.
                                Defaults to all teams in team_season_dict.

    Returns:
        dict mapping team_name -> {
            "seasons":               [str, ...]        season tokens in order
            "ts_embeddings":         (n_seasons, D)    team-season embeddings
            "combined_embeddings":   (n_seasons, D)    team + team-season embeddings
            "ts_displacements":      [float, ...]      cosine distances between
                                                       consecutive team-season embeddings
            "combined_displacements":[float, ...]      same for combined embeddings
            "n_seasons":             int
        }
        Displacement lists have length n_seasons - 1.
    """
    teams = teams or list(team_season_dict.keys())
    team_name_to_id = {name: info["team_id"] for name, info in team_season_dict.items()}
    results = {}

    for team_name in teams:
        info = team_season_dict.get(team_name)
        if info is None:
            print(f"[WARN] '{team_name}' not found in team_season_dict, skipping.")
            continue
        if info["n_seasons"] < 2:
            continue

        indices = info["season_indices"]
        seasons = info["seasons"]

        ts_embs = np.stack(
            [embedding_matrix[i] for i in indices], axis=0
        ).astype(np.float32)  # (n_seasons, D)

        team_id = team_name_to_id.get(team_name)
        static  = static_team_embeddings.get(team_id)
        if static is None:
            print(f"[WARN] No static embedding for '{team_name}' (id={team_id}), skipping.")
            continue

        if hasattr(static, "numpy"):
            static = static.numpy()
        combined_embs = ts_embs + static.astype(np.float32)  # (n_seasons, D)

        ts_norm       = normalize(ts_embs,       norm="l2")
        combined_norm = normalize(combined_embs, norm="l2")

        ts_displacements       = [
            float(1.0 - ts_norm[i] @ ts_norm[i + 1])
            for i in range(len(seasons) - 1)
        ]
        combined_displacements = [
            float(1.0 - combined_norm[i] @ combined_norm[i + 1])
            for i in range(len(seasons) - 1)
        ]

        results[team_name] = {
            "seasons":                seasons,
            "ts_embeddings":          ts_embs,
            "combined_embeddings":    combined_embs,
            "ts_displacements":       ts_displacements,
            "combined_displacements": combined_displacements,
            "n_seasons":              len(seasons),
        }

    return results


# -----------------------------------------------------------------------------
# Figures
# -----------------------------------------------------------------------------

def _extract_year(season_token: str) -> int:
    """Extract the start year from a season token like '2015_2016'."""
    return int(season_token.split("_")[0])


def _build_canonical_transitions(trajectories: Dict) -> List[str]:
    """
    Build the canonical list of year-to-year transitions covering all seasons
    present across all teams, e.g. ['2015_2016 → 2016_2017', '2016_2017 → 2017_2018', ...]
    """
    all_years = set()
    for info in trajectories.values():
        for s in info["seasons"]:
            all_years.add(_extract_year(s))
    sorted_years = sorted(all_years)
    return [
        f"{y}_{y+1} → {y+1}_{y+2}"
        for y in sorted_years[:-1]
    ]


def _build_gap_aware_displacement(
    info: Dict,
    disp_key: str,
    canonical_transitions: List[str],
    embedding_matrix: np.ndarray,
    static_emb: Optional[np.ndarray],
    mode: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a single team, map displacements onto canonical year-to-year transitions.
    Across-gap transitions get the distance between the last pre-gap and first
    post-gap embedding, placed in the bucket of the transition into the return season.

    Returns:
        values:   (n_transitions,) float array, nan where team absent both sides
        is_gap:   (n_transitions,) bool array, True where value spans a gap
    """
    seasons    = info["seasons"]
    season_set = set(seasons)
    season_to_idx = {s: i for i, s in enumerate(seasons)}

    n = len(canonical_transitions)
    values = np.full(n, np.nan)
    is_gap = np.zeros(n, dtype=bool)

    ts_embs      = info["ts_embeddings"]
    combined_embs = info["combined_embeddings"]
    embs = ts_embs if mode == "ts" else combined_embs

    from sklearn.preprocessing import normalize as sk_normalize
    embs_norm = sk_normalize(embs.astype(np.float32), norm="l2")

    for col, trans in enumerate(canonical_transitions):
        left_s, right_s = trans.split(" → ")

        if left_s in season_set and right_s in season_set:
            # Consecutive seasons both present — standard transition
            li = season_to_idx[left_s]
            ri = season_to_idx[right_s]
            values[col] = float(1.0 - embs_norm[li] @ embs_norm[ri])
            is_gap[col] = False

        elif right_s in season_set:
            # Right season present but left is not — find most recent prior season
            right_year = _extract_year(right_s)
            prior_seasons = [s for s in seasons if _extract_year(s) < right_year]
            if prior_seasons:
                last_prior = max(prior_seasons, key=_extract_year)
                li = season_to_idx[last_prior]
                ri = season_to_idx[right_s]
                values[col] = float(1.0 - embs_norm[li] @ embs_norm[ri])
                is_gap[col] = True  # spans a gap

    return values, is_gap


def plot_displacement_heatmap(
    trajectories: Dict[str, Dict],
    mode: str = "ts",
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
) -> Figure:
    """
    Heatmap of season-to-season cosine displacement for all teams.
    Rows = teams, columns = canonical year-to-year transitions.
    Missing seasons (team absent both sides of transition) are grey.
    Across-gap transitions are hatched to distinguish them from true
    year-to-year displacements.

    Args:
        trajectories: output of compute_trajectories()
        mode:         "ts" for team-season only, "combined" for team + team-season
        title:        plot title (auto-generated if None)
        figsize:      figure size
    """
    assert mode in ("ts", "combined"), "mode must be 'ts' or 'combined'"
    disp_key = "ts_displacements" if mode == "ts" else "combined_displacements"

    canonical = _build_canonical_transitions(trajectories)
    team_names = sorted(trajectories.keys())

    matrix  = np.full((len(team_names), len(canonical)), np.nan)
    gap_mask = np.zeros((len(team_names), len(canonical)), dtype=bool)

    for row, team_name in enumerate(team_names):
        info = trajectories[team_name]
        values, is_gap = _build_gap_aware_displacement(
            info, disp_key, canonical,
            embedding_matrix=None,  # not needed, embeddings already in info
            static_emb=None,
            mode=mode,
        )
        matrix[row]   = values
        gap_mask[row] = is_gap

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#dddddd")

    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="Cosine distance", fraction=0.03, pad=0.02)

    # Hatch gap cells
    for row in range(len(team_names)):
        for col in range(len(canonical)):
            if gap_mask[row, col]:
                ax.add_patch(plt.Rectangle(
                    (col - 0.5, row - 0.5), 1, 1,
                    fill=False, hatch="///", edgecolor="white",
                    linewidth=0, alpha=0.6,
                ))

    ax.set_yticks(range(len(team_names)))
    ax.set_yticklabels(team_names, fontsize=7)
    ax.set_xticks(range(len(canonical)))
    ax.set_xticklabels(canonical, rotation=45, ha="right", fontsize=7)

    label = "team-season only" if mode == "ts" else "team + team-season"
    ax.set_title(title or f"Season-to-season displacement ({label})")

    # Legend for hatch
    from matplotlib.patches import Patch
    ax.legend(
        handles=[Patch(facecolor="white", edgecolor="grey", hatch="///", label="across gap")],
        loc="lower right", fontsize=7,
    )

    fig.tight_layout()
    return fig


def plot_displacement_lines(
    trajectories: Dict[str, Dict],
    mode: str = "ts",
    highlight_teams: Optional[List[str]] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (13, 6),
) -> Figure:
    """
    Line plot of season-to-season cosine displacement over time per team.
    All teams are shown in grey; highlight_teams are shown in colour on top.

    Args:
        trajectories:    output of compute_trajectories()
        mode:            "ts" or "combined"
        highlight_teams: teams to draw in colour (others in grey)
        title:           plot title
        figsize:         figure size
    """
    assert mode in ("ts", "combined"), "mode must be 'ts' or 'combined'"
    disp_key = "ts_displacements" if mode == "ts" else "combined_displacements"

    fig, ax = plt.subplots(figsize=figsize)
    highlight_set = set(highlight_teams or [])

    canonical  = _build_canonical_transitions(trajectories)
    trans_idx  = {t: i for i, t in enumerate(canonical)}

    colours = plt.cm.get_cmap("tab10", max(len(highlight_set), 1))
    colour_map = {t: colours(i) for i, t in enumerate(sorted(highlight_set))}

    # Grey background lines first
    for team_name, info in trajectories.items():
        if team_name in highlight_set:
            continue
        values, _ = _build_gap_aware_displacement(
            info, disp_key, canonical, None, None, mode
        )
        valid = [(i, v) for i, v in enumerate(values) if not np.isnan(v)]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, color="#cccccc", linewidth=0.8, alpha=0.5, zorder=1)

    # Highlighted lines on top
    for team_name in sorted(highlight_set):
        info = trajectories.get(team_name)
        if info is None:
            continue
        values, is_gap = _build_gap_aware_displacement(
            info, disp_key, canonical, None, None, mode
        )
        valid = [(i, v, is_gap[i]) for i, v in enumerate(values) if not np.isnan(v)]
        if not valid:
            continue
        xs, ys, gaps = zip(*valid)
        # Solid line for normal, dashed for gap transitions
        for j in range(len(xs)):
            style = "--" if gaps[j] else "-"
            if j < len(xs) - 1:
                ax.plot(xs[j:j+2], ys[j:j+2], color=colour_map[team_name],
                        linewidth=2.0, linestyle=style, zorder=2)
        ax.scatter(xs, ys, color=colour_map[team_name], s=20,
                   marker="o", zorder=3, label=team_name)

    ax.set_xticks(range(len(canonical)))
    ax.set_xticklabels(canonical, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Cosine distance", fontsize=10)

    label = "team-season only" if mode == "ts" else "team + team-season"
    ax.set_title(title or f"Season-to-season displacement ({label})")

    if highlight_set:
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)

    fig.tight_layout()
    return fig


def plot_displacement_comparison(
    trajectories: Dict[str, Dict],
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 7),
) -> Figure:
    """
    Scatter plot comparing ts-only vs combined displacement for each
    team-season transition. Points above the diagonal indicate the combined
    representation moves more than the team-season alone.

    Args:
        trajectories: output of compute_trajectories()
        title:        plot title
        figsize:      figure size
    """
    ts_vals, combined_vals, labels = [], [], []

    for team_name, info in trajectories.items():
        for i, (ts_d, comb_d) in enumerate(
            zip(info["ts_displacements"], info["combined_displacements"])
        ):
            ts_vals.append(ts_d)
            combined_vals.append(comb_d)
            labels.append(f"{team_name}\n{info['seasons'][i]}→{info['seasons'][i+1]}")

    ts_vals      = np.array(ts_vals)
    combined_vals = np.array(combined_vals)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(ts_vals, combined_vals, s=20, alpha=0.5, linewidths=0, color="steelblue")

    lim = max(ts_vals.max(), combined_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, label="y = x")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

    ax.set_xlabel("Team-season displacement", fontsize=10)
    ax.set_ylabel("Combined displacement", fontsize=10)
    ax.set_title(title or "Displacement: team-season vs combined")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------

def run_trajectory_analysis(
    team_season_dict: Dict,
    embedding_matrix: np.ndarray,
    static_team_embeddings: Dict[str, np.ndarray],
    teams: Optional[List[str]] = None,
    highlight_teams: Optional[List[str]] = None,
) -> Tuple[Dict, Dict[str, Figure]]:
    """
    Run full trajectory analysis and return data + all figures.

    Returns:
        trajectories: dict from compute_trajectories()
        figures: {
            "heatmap_ts":       displacement heatmap, team-season only
            "heatmap_combined": displacement heatmap, combined
            "lines_ts":         line plot, team-season only
            "lines_combined":   line plot, combined
            "comparison":       ts vs combined scatter
        }
    """
    trajectories = compute_trajectories(
        team_season_dict, embedding_matrix, static_team_embeddings, teams
    )

    figures = {
        "heatmap_ts":       plot_displacement_heatmap(trajectories, mode="ts"),
        "heatmap_combined": plot_displacement_heatmap(trajectories, mode="combined"),
        "lines_ts":         plot_displacement_lines(trajectories, mode="ts",
                                highlight_teams=highlight_teams),
        "lines_combined":   plot_displacement_lines(trajectories, mode="combined",
                                highlight_teams=highlight_teams),
        "comparison":       plot_displacement_comparison(trajectories),
    }

    return trajectories, figures

def rank_displacements(
    trajectories: Dict,
    mode: str = "ts",
    n: int = 20,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Rank all team-season transitions by cosine displacement.

    Args:
        trajectories: output of compute_trajectories()
        mode:         "ts" or "combined"
        n:            number of top/bottom transitions to return

    Returns:
        top_n:    DataFrame of the n highest displacement transitions
        bottom_n: DataFrame of the n lowest displacement transitions
        Both have columns: team, from_season, to_season, displacement, is_gap
    """
    canonical = _build_canonical_transitions(trajectories)
    records = []

    for team_name, info in trajectories.items():
        values, is_gap = _build_gap_aware_displacement(
            info, f"{mode}_displacements", canonical, None, None, mode
        )
        for col, (v, gap) in enumerate(zip(values, is_gap)):
            if np.isnan(v):
                continue
            left_s, right_s = canonical[col].split(" → ")
            records.append({
                "team":         team_name,
                "from_season":  left_s,
                "to_season":    right_s,
                "displacement": round(float(v), 4),
                "is_gap":       bool(gap),
            })

    df = pd.DataFrame(records).sort_values("displacement", ascending=False)
    return df.head(n).reset_index(drop=True), df.tail(n).reset_index(drop=True)

# -----------------------------------------------------------------------------
# Build flat embedding dict for a given league's trajectories
# -----------------------------------------------------------------------------

def build_ts_embeds(trajectories, embedding_matrix):
    """
    Build flat dicts of team-season embeddings for use with existing
    cos_sim_mat, pca, and umap_embed functions.

    Returns:
        embeds_dict:  {"{team}_{season}": np.ndarray (D,)}
        team_attr:    {"{team}_{season}": team_name}   for coloring by team
        season_attr:  {"{team}_{season}": season}      for coloring by season
    """
    embeds_dict  = {}
    team_attr    = {}
    season_attr  = {}

    for team_name, info in trajectories.items():
        for season, embed in zip(info["seasons"], info["ts_embeddings"]):
            key = f"{team_name}_{season}"
            embeds_dict[key] = embed
            team_attr[key]   = team_name
            season_attr[key] = season

    return embeds_dict, team_attr, season_attr

# -----------------------------------------------------------------------------
# Side-by-side similarity matrices: sorted by team vs sorted by season
# -----------------------------------------------------------------------------

def plot_team_vs_season_similarity(
    embeds_dict: Dict,
    team_attr: Dict,
    season_attr: Dict,
    title_prefix: str = "",
    figsize: Tuple[int, int] = (18, 8),
) -> Figure:
    """
    Two side-by-side cosine similarity heatmaps:
        Left:  points sorted by team   — strong block structure = team signal dominates
        Right: points sorted by season — strong block structure = era signal dominates

    Args:
        embeds_dict:  output of build_ts_embeds()
        team_attr:    output of build_ts_embeds()
        season_attr:  output of build_ts_embeds()
        title_prefix: prepended to each subplot title
        figsize:      figure size
    """
    # Sort by team
    sim_team, ids_team, attrs_team = cos_sim_mat(embeds_dict, attribute_dict=team_attr)
    # Sort by season
    sim_season, ids_season, attrs_season = cos_sim_mat(embeds_dict, attribute_dict=season_attr)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, sim, ids, attrs, sort_label in [
        (axes[0], sim_team,   ids_team,   attrs_team,   "team"),
        (axes[1], sim_season, ids_season, attrs_season, "season"),
    ]:
        cmap = plt.cm.RdBu_r.copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(sim, vmin=-1, vmax=1, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Colour tick labels by attribute
        unique_attrs = sorted(set(a for a in attrs if a is not None))
        cmap_attr    = plt.cm.get_cmap("tab20", len(unique_attrs))
        colour_map   = {a: cmap_attr(i) for i, a in enumerate(unique_attrs)}

        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels([str(i).split("_")[0] for i in ids],  # just team name prefix
                           rotation=90, fontsize=4)
        ax.set_yticklabels([str(i).split("_")[0] for i in ids], fontsize=4)

        for tick, attr in zip(ax.get_xticklabels(), attrs):
            tick.set_color(colour_map.get(attr, (0, 0, 0, 1)))
        for tick, attr in zip(ax.get_yticklabels(), attrs):
            tick.set_color(colour_map.get(attr, (0, 0, 0, 1)))

        # Block boundary lines
        for i in range(len(attrs) - 1):
            if attrs[i] != attrs[i + 1]:
                ax.axhline(i + 0.5, color="black", linewidth=0.6, linestyle="--")
                ax.axvline(i + 0.5, color="black", linewidth=0.6, linestyle="--")

        prefix = f"{title_prefix} — " if title_prefix else ""
        ax.set_title(f"{prefix}sorted by {sort_label}")

    fig.tight_layout()
    return fig


# -----------------------------------------------------------------------------
# Block coherence score
# -----------------------------------------------------------------------------

def block_coherence(sim_matrix: np.ndarray, attrs: List[str]) -> float:
    """
    Compute mean within-block similarity minus mean cross-block similarity.
    Higher = stronger block structure for the given attribute grouping.
    """
    attrs_arr = np.array(attrs)
    within, cross = [], []
    n = len(attrs)
    for i in range(n):
        for j in range(i + 1, n):
            if attrs_arr[i] == attrs_arr[j]:
                within.append(sim_matrix[i, j])
            else:
                cross.append(sim_matrix[i, j])
    return float(np.mean(within) - np.mean(cross))

def get_primary_affiliation(player,attribute_dict,position_cats=None):
    all_affiliations = attribute_dict[player]
    primary_affiliation = max(all_affiliations, key=lambda x: x[1])[0]
    if position_cats != None:
        return position_cats[primary_affiliation]
    else:
        return primary_affiliation
    
def league_to_sex(league):
    womens = {"England - FA Women's Super League", "Germany - Frauen Bundesliga",
              "Italy - Serie A Women", "Spain - Liga F", 
              "United States of America - NWSL"}
    return "women" if league in womens else "men"


# =============================================================================
# In-context layered embedding analysis helpers
# =============================================================================

from scipy.stats import spearmanr as _spearmanr
from scipy.spatial.distance import squareform as _squareform
from sklearn.metrics import balanced_accuracy_score as _balanced_accuracy_score


def block_coherence_per_layer(
    mat: np.ndarray,
    group_labels: List,
) -> np.ndarray:
    """
    Compute block coherence at each transformer layer: mean within-group cosine
    similarity minus mean between-group cosine similarity.

    Positive values → the model represents members of the same group more
    similarly than members of different groups. 0 = no group structure.

    Args:
        mat:          (N, N_layers, d) embedding array
        group_labels: list of N group identifiers (e.g. team IDs)

    Returns:
        (N_layers,) array of coherence scores (approx. in [-1, 1])
    """
    N, N_layers, d = mat.shape
    labels = np.array(group_labels)
    same  = labels[:, None] == labels[None, :]          # (N, N) bool
    upper = np.triu(np.ones((N, N), dtype=bool), k=1)
    within_mask  = same  & upper
    between_mask = ~same & upper

    coherence = np.zeros(N_layers)
    for l in range(N_layers):
        vecs  = mat[:, l, :].astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_n = vecs / np.where(norms > 0, norms, 1.0)
        sim = vecs_n @ vecs_n.T
        w = sim[within_mask].mean()  if within_mask.any()  else 0.0
        b = sim[between_mask].mean() if between_mask.any() else 0.0
        coherence[l] = w - b
    return coherence


def linear_probe_per_layer(
    mat: np.ndarray,
    labels: List,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Fit a logistic regression probe at each transformer layer and return the
    stratified k-fold cross-validated accuracy.

    Embeddings are standardised and reduced to at most 50 PCA components before
    probing to avoid the curse of dimensionality with small N.

    Args:
        mat:          (N, N_layers, d) embedding array
        labels:       list of N class labels (e.g. team IDs)
        n_folds:      number of CV folds (clamped to min samples per class)
        random_state: random seed

    Returns:
        (N_layers,) array of CV accuracies in [0, 1], or NaN if not computable
    """
    N, N_layers, d = mat.shape
    le = LabelEncoder()
    y  = le.fit_transform(labels)
    min_per_class = int(np.bincount(y).min())
    k  = min(n_folds, min_per_class)
    if k < 2:
        return np.full(N_layers, np.nan)

    cv   = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    accs = np.zeros(N_layers)
    for l in range(N_layers):
        X      = StandardScaler().fit_transform(mat[:, l, :])
        n_comp = min(50, X.shape[1], N - 1)
        X_pca  = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
        clf    = LogisticRegression(
            max_iter=500, random_state=random_state,
            C=1.0, solver="lbfgs", multi_class="multinomial",
        )
        y_pred    = cross_val_predict(clf, X_pca, y, cv=cv)
        accs[l]   = accuracy_score(y, y_pred)
    return accs


def effective_rank_per_layer(mat: np.ndarray) -> np.ndarray:
    """
    Compute the effective rank (participation ratio) of the embedding cloud at
    each transformer layer.

        PR = (Σ σ_i)² / Σ σ_i²

    Equals 1 when a single direction dominates; equals min(N, d) when all
    directions contribute equally.

    Args:
        mat: (N, N_layers, d) embedding array

    Returns:
        (N_layers,) array of effective ranks
    """
    N, N_layers, d = mat.shape
    ranks = np.zeros(N_layers)
    for l in range(N_layers):
        vecs = mat[:, l, :].astype(np.float64)
        vecs -= vecs.mean(axis=0)
        _, sv, _ = np.linalg.svd(vecs, full_matrices=False)
        sv = sv[sv > 0]
        if sv.size > 0:
            ranks[l] = sv.sum() ** 2 / (sv ** 2).sum()
    return ranks


def token_discriminability_per_layer(
    mat1: np.ndarray,
    mat2: np.ndarray,
    n_folds: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    At each transformer layer, measure how well a binary logistic regression
    classifier can distinguish two token types. Returns balanced accuracy
    (0.5 = chance, 1.0 = perfect separation).

    Args:
        mat1:         (N1, N_layers, d) embeddings for token type 1
        mat2:         (N2, N_layers, d) embeddings for token type 2
        n_folds:      CV folds (clamped to min class size)
        random_state: random seed

    Returns:
        (N_layers,) array of balanced accuracies
    """
    N1, N_layers, d = mat1.shape
    N2 = mat2.shape[0]
    y  = np.array([0] * N1 + [1] * N2)
    k  = min(n_folds, N1, N2)
    if k < 2:
        return np.full(N_layers, np.nan)

    cv   = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)
    accs = np.zeros(N_layers)
    for l in range(N_layers):
        X      = np.vstack([mat1[:, l, :], mat2[:, l, :]])
        X      = StandardScaler().fit_transform(X)
        n_comp = min(50, X.shape[1], len(X) - 1)
        X_pca  = PCA(n_components=n_comp, random_state=random_state).fit_transform(X)
        clf    = LogisticRegression(max_iter=500, random_state=random_state, C=1.0)
        y_pred = cross_val_predict(clf, X_pca, y, cv=cv)
        accs[l] = _balanced_accuracy_score(y, y_pred)
    return accs


def pls_label_r2_per_layer(
    mat: np.ndarray,
    labels: List,
    n_components: int = 3,
) -> np.ndarray:
    """
    At each transformer layer, fit PLS-DA (PLS regression onto one-hot encoded
    labels) and return the in-sample R² of the label matrix.

    R² measures how much of the label variance can be reconstructed from the
    first `n_components` PLS directions of the embedding. Higher R² at a layer
    means more label information is linearly accessible.

    Note: this is training R², not cross-validated. Use it for relative
    comparisons across layers, not as an absolute measure of generalisation.

    Args:
        mat:          (N, N_layers, d) embedding array
        labels:       list of N class labels
        n_components: number of PLS components (clamped to valid range)

    Returns:
        (N_layers,) array of R² values in [0, 1]
    """
    N, N_layers, d = mat.shape
    le  = LabelEncoder()
    y_int = le.fit_transform(labels)
    ohe = OneHotEncoder(sparse_output=False)
    Y   = ohe.fit_transform(y_int.reshape(-1, 1))   # (N, n_classes)
    ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)

    r2s = np.zeros(N_layers)
    for l in range(N_layers):
        X      = StandardScaler().fit_transform(mat[:, l, :])
        n_comp = min(n_components, len(le.classes_) - 1, N - 1)
        if n_comp < 1:
            continue
        pls    = PLSRegression(n_components=n_comp, scale=False)
        pls.fit(X, Y)
        Y_pred = pls.predict(X)
        ss_res = np.sum((Y - Y_pred) ** 2)
        r2s[l] = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return r2s


def rsa_matrix_per_layer(mat: np.ndarray) -> np.ndarray:
    """
    Compute the Spearman correlation between every pair of layers' RDMs
    (Representational Dissimilarity Matrices).

    A high correlation between layers i and j means the geometry of
    inter-entity differences is preserved across those layers.

    Args:
        mat: (N, N_layers, d) embedding array

    Returns:
        (N_layers, N_layers) Spearman correlation matrix
    """
    N, N_layers, d = mat.shape

    def _rdm_vec(l):
        vecs  = mat[:, l, :].astype(np.float64)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs_n = vecs / np.where(norms > 0, norms, 1.0)
        D = 1.0 - vecs_n @ vecs_n.T   # cosine distance matrix
        np.fill_diagonal(D, 0.0)       # clamp FP noise on diagonal
        return _squareform(D)          # upper-triangle flat vector

    rdm_vecs = [_rdm_vec(l) for l in range(N_layers)]
    R = np.zeros((N_layers, N_layers))
    for i in range(N_layers):
        for j in range(N_layers):
            r, _ = _spearmanr(rdm_vecs[i], rdm_vecs[j])
            R[i, j] = r
    return R

