import json
from collections import Counter
from pathlib import Path
from typing import Any, Hashable, Optional

import numpy as np


EmbeddingDict = dict[int, np.ndarray]
AttributeDict = dict[int, str]
RGBAColor = tuple[float, float, float, float]


def align_static_audit_data(embeds_dict: EmbeddingDict, attr_dict: AttributeDict,
                            min_samples: int = 5) -> tuple[EmbeddingDict, AttributeDict, dict[str, Any]]:
    """
    Align static embeddings and attributes by shared ids.

    Drops ids with missing labels, then drops classes with fewer than
    min_samples examples. Returns filtered dictionaries plus metadata about
    what was kept.
    :param embeds_dict: embedding dictionary of key as team id and value as the vector
    :param attr_dict: attribute dictionary of key as team id and value as the string classification
    :param min_samples: minimum number of examples containing attribute to keep (helps filter out low data that
    could crash future analysis)
    :return tuple[EmbeddingDict, AttributeDict, dict[str, Any]] (linked embedding attribute data that is filtered
    with minor data pre-processing
    """

    # list comprehension to filter out team_ids that don't have corresponding metadata
    ids = [item_id for item_id in embeds_dict if item_id in attr_dict and attr_dict[item_id] is not None]
    # use counter to get counts of each categorical metadata entry
    label_counts = Counter(attr_dict[item_id] for item_id in ids)
    # discard attribute category if it has less than 5 occurances
    kept_labels = {label for label, count in label_counts.items() if count >= min_samples}
    # make a list of ids with low occurance ids filtered out
    kept_ids = [item_id for item_id in ids if attr_dict[item_id] in kept_labels]

    # filter out the dicitonaries using only ids from the kept_ids list
    filtered_embeds = {item_id: embeds_dict[item_id] for item_id in kept_ids}
    filtered_attrs = {item_id: attr_dict[item_id] for item_id in kept_ids}
    filtered_counts = Counter(filtered_attrs.values())


    # summary metadata dictionary that provides information on the filtering
    metadata = {
        "n_overlap_before_filter": len(ids),
        "n_items": len(kept_ids),
        "n_classes": len(filtered_counts),
        "class_counts": dict(sorted(filtered_counts.items())),
        # sort and iterate through dropped labels
        "dropped_class_counts": {label: count for label, count in sorted(label_counts.items())
                                 if label not in kept_labels},
    }
    return filtered_embeds, filtered_attrs, metadata


def cos_sim_mat(embeds_dict: EmbeddingDict, attribute_dict: Optional[AttributeDict] = None) \
        -> tuple[np.ndarray, list[Hashable], Optional[list[str]]]:
    """
    Compute pairwise cosine similarity for id -> vector dictionaries.

    If attribute_dict is provided, ids are sorted by attribute label first so
    same-label examples form contiguous blocks in heatmaps.

    :param embeds_dict: embedding dictionary with key as team_id and value as the vector
    :param attribute_dict: attribute dictionary with key as team id and value as the string classification
    :return tuple[np.ndarray, list[Hashable], Optional[list[str]]]
    These are the N by N similarity matrix, the list of ids in the same order as the matrix which are also dictionary
    keys for other dictionaries and the list of attributes for the ids
    """

    ids = list(embeds_dict)
    if attribute_dict is not None:
        ids.sort(key=lambda item_id: (attribute_dict.get(item_id, ""), item_id))
        attributes = [attribute_dict.get(item_id) for item_id in ids]
    else:
        attributes = None

    # turn embeddings into a 2D matrix where each row is an embedding vector in the sorted order
    vecs = np.stack([embeds_dict[item_id] for item_id in ids], axis=0)
    # find the length of each vector by computing length = sqrt(x1^2 + x2^2 ... xi^2)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    # normalize the vectors by doing (x1/length, x2/length ... xi/length). Results in length being 1
    # if vector is 0, then divide by 1 instead of norms to avoid divisionbyzerro error
    vecs = vecs / np.where(norms > 0, norms, 1.0)
    # cosine sim: dot(a, b) / (length(a) * length(b)) = dot(a, b) / (1 * 1)
    # with normalization becomes dot(a, b)
    sim_matrix = vecs @ vecs.T

    return sim_matrix, ids, attributes


def _make_attribute_colormap(attributes: list[str]) -> tuple[dict[str, RGBAColor], list[str]]:
    """
    Generate a colormap dictionary for attribute type
    :param attributes: list of attribute strings
    :return tuple[dict[str, RGBAColor], list[str]]
    Dictionary of key string from attribute and value the cmap tuple
    List of strings of the attributes used in the respective order
    """
    # import matplotlib locally in the function
    import matplotlib

    # use agg to prevent opening figures on the cluster and just save them
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # get a sorted list of unique not none attributes
    unique_attrs = sorted(set(attr for attr in attributes if attr is not None))
    # generate a color palette based on the unique attributes length
    cmap = plt.colormaps.get_cmap("tab20").resampled(len(unique_attrs))
    # dictionary generation for attribute and its color type
    color_map = {}
    for i, label in enumerate(unique_attrs):
        red, green, blue, alpha = cmap(i)
        color_map[label] = (float(red), float(green), float(blue), float(alpha))
    return color_map, unique_attrs


def plot_heatmap(sim_matrix: np.ndarray, ids: list[Hashable], attributes: Optional[list[str]] = None,
    title: str = "Cosine Similarity Matrix", figsize: tuple[int, int] = (12, 10)):
    """
    Plot a cosine similarity heatmap with optional attribute block boundaries.
    """

    # import matplot inside the function and use Agg for non-interactive plotting
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    # create the figure and one axis
    fig, ax = plt.subplots(figsize=figsize)
    # draw the heatmap, set color scale to -1 to 1
    # use red/blue coloring and let matplotlib stretch image to fit the figure
    image = ax.imshow(sim_matrix, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    # Add a color bar next to the heatmap
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    # Create a tick position for every row/column
    ax.set_xticks(range(len(ids)))
    ax.set_yticks(range(len(ids)))
    # covert IDs to strings and add them to the x and y labels for their corresponding row/column
    ax.set_xticklabels([str(item_id) for item_id in ids], rotation=90, fontsize=6)
    ax.set_yticklabels([str(item_id) for item_id in ids], fontsize=6)

    # only do attribute label coloring if attributes were provided
    if attributes is not None:
        # use helper function ot generate the colormap for the attributes
        color_map, unique_attrs = _make_attribute_colormap(attributes)

        # loop through matched ticks and attributes and color the tick based off of the color map's results
        # if the attribute doesn't have a color, default to black
        for tick, attr in zip(ax.get_xticklabels(), attributes):
            tick.set_color(color_map.get(attr, (0.0, 0.0, 0.0, 1.0)))
        for tick, attr in zip(ax.get_yticklabels(), attributes):
            tick.set_color(color_map.get(attr, (0.0, 0.0, 0.0, 1.0)))

        # loop through the length of the attributes, and if an attribute is different than the next one, create
        # a boundary at i + .5, between the two different ones
        boundaries = [i + 0.5 for i in range(len(attributes) - 1) if attributes[i] != attributes[i + 1]]

        # for each of those boundaries, add a line between them to symbolize attribute blocks
        for boundary in boundaries:
            ax.axhline(boundary, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(boundary, color="black", linewidth=0.8, linestyle="--")

        # list comprehension for color patches for each attribute
        handles = [mpatches.Patch(color=color_map[attr], label=attr) for attr in unique_attrs]

        # add a legend to the plot with handle color patches as handles
        # place the legend relative to the axis box x=1.15 slightly to the right of the plot and y=1 at the top
        # in the upper left with fontsize 7 and opacity mostly opaque at .8
        ax.legend(handles=handles, bbox_to_anchor=(1.15, 1), loc="upper left", fontsize=7, framealpha=0.8)

    # set the title and adjust spacing so everything fits better
    ax.set_title(title)
    fig.tight_layout()

    # return the figure back to the caller to decide what to do next
    return fig


def block_coherence(sim_matrix: np.ndarray, attrs: list[str]) -> float:
    """
    Return mean same-label similarity minus mean cross-label similarity.
    """
    attrs_arr = np.array(attrs)
    same_label = attrs_arr[:, None] == attrs_arr[None, :]
    upper = np.triu(np.ones(sim_matrix.shape, dtype=bool), k=1)
    within = sim_matrix[same_label & upper]
    cross = sim_matrix[~same_label & upper]

    if within.size == 0 or cross.size == 0:
        return float("nan")
    return float(np.mean(within) - np.mean(cross))


def logistic_probe(
    embeds_dict: EmbeddingDict,
    attr_dict: AttributeDict,
    n_folds: int = 5,
    max_iter: int = 1000,
    random_state: int = 1337,
) -> dict[str, Any]:
    """
    Generic multinomial logistic regression probe for static embeddings.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    ids = [item_id for item_id in embeds_dict if item_id in attr_dict]
    if not ids:
        raise ValueError("No overlap between embeds_dict and attr_dict keys.")

    labels = [attr_dict[item_id] for item_id in ids]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    class_counts = np.bincount(y)
    if len(class_counts) < 2:
        raise ValueError("Logistic probe requires at least two classes.")

    effective_folds = min(n_folds, int(class_counts.min()))
    if effective_folds < 2:
        raise ValueError("Logistic probe requires at least two samples per class.")

    X = np.stack([embeds_dict[item_id] for item_id in ids], axis=0).astype(np.float32)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
    )

    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=random_state,
    )
    y_pred_cv = cross_val_predict(clf, X_scaled, y, cv=cv)

    fold_accuracies = []
    for train_idx, val_idx in cv.split(X_scaled, y):
        clf_fold = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
        )
        clf_fold.fit(X_scaled[train_idx], y[train_idx])
        fold_accuracies.append(
            accuracy_score(y[val_idx], clf_fold.predict(X_scaled[val_idx]))
        )

    clf.fit(X_scaled, y)
    y_true_labels = le.inverse_transform(y)
    y_pred_labels = le.inverse_transform(y_pred_cv)

    per_class_accuracy = {}
    for label in le.classes_:
        mask = y_true_labels == label
        per_class_accuracy[label] = accuracy_score(
            y_true_labels[mask],
            y_pred_labels[mask],
        )

    return {
        "model": clf,
        "scaler": scaler,
        "label_encoder": le,
        "ids": ids,
        "y_true": y_true_labels,
        "y_pred_cv": y_pred_labels,
        "overall_accuracy": accuracy_score(y, y_pred_cv),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": confusion_matrix(
            y,
            y_pred_cv,
            labels=list(range(len(le.classes_))),
        ),
        "class_names": list(le.classes_),
        "classification_report": classification_report(y_true_labels, y_pred_labels),
        "cv_fold_accuracies": fold_accuracies,
        "coef": clf.coef_,
        "n_folds_used": effective_folds,
    }


def pca_component_probe(
    embeds_dict: EmbeddingDict,
    attr_dict: AttributeDict,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = 0.95,
    alpha: float = 0.05,
    random_state: int = 1337,
) -> dict[str, Any]:
    """
    Test whether PCA components encode attribute labels using MNLogit.
    """
    import pandas as pd
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from statsmodels.discrete.discrete_model import MNLogit
    from statsmodels.stats.multitest import multipletests

    ids = [item_id for item_id in embeds_dict if item_id in attr_dict]
    if not ids:
        raise ValueError("No overlap between embeds_dict and attr_dict keys.")

    labels = [attr_dict[item_id] for item_id in ids]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    if len(le.classes_) < 2:
        raise ValueError("PCA component probe requires at least two classes.")

    X = np.stack([embeds_dict[item_id] for item_id in ids], axis=0).astype(np.float64)
    X_scaled = StandardScaler().fit_transform(X)

    max_components = min(len(ids), X_scaled.shape[1])
    pca_full = PCA(n_components=max_components, random_state=random_state)
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    if n_components is not None:
        n_used = min(n_components, max_components)
    else:
        n_used = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_used = min(n_used, max_components)

    pca = PCA(n_components=n_used, random_state=random_state)
    coords = pca.fit_transform(X_scaled)

    X_mnlogit = np.hstack([np.ones((len(ids), 1)), coords])
    model = MNLogit(y, X_mnlogit)
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)

    pvals_per_component = result.pvalues[1:, :]
    pvals_min = pvals_per_component.min(axis=1)
    reject, pvals_corrected, _, _ = multipletests(
        pvals_min,
        alpha=alpha,
        method="fdr_bh",
    )
    significant_components = [idx for idx, is_sig in enumerate(reject) if is_sig]

    coefs = result.params[1:, :]
    max_coef_labels = []
    for idx in range(n_used):
        class_idx = int(np.abs(coefs[idx]).argmax())
        label_idx = min(class_idx + 1, len(le.classes_) - 1)
        max_coef_labels.append(le.classes_[label_idx])

    results_table = pd.DataFrame(
        {
            "component": range(n_used),
            "explained_var": pca.explained_variance_ratio_,
            "cumulative_var": np.cumsum(pca.explained_variance_ratio_),
            "p_value_raw": pvals_min,
            "p_value_corrected": pvals_corrected,
            "significant": reject,
            "max_coef_label": max_coef_labels,
        }
    )

    return {
        "pca": pca,
        "pca_coords": coords,
        "n_components_used": n_used,
        "explained_variance": pca.explained_variance_ratio_,
        "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        "label_encoder": le,
        "ids": ids,
        "results_table": results_table,
        "significant_components": significant_components,
        "loadings": pca.components_,
        "mnlogit_result": result,
    }


def pls_probe(
    embeds_dict: EmbeddingDict,
    attr_dict: AttributeDict,
    n_components: Optional[int] = None,
    alpha: float = 0.05,
    random_state: int = 1337,
) -> dict[str, Any]:
    """
    Test whether PLS components encode attribute labels using MNLogit.
    """
    import pandas as pd
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
    from statsmodels.discrete.discrete_model import MNLogit
    from statsmodels.stats.multitest import multipletests

    ids = [item_id for item_id in embeds_dict if item_id in attr_dict]
    if not ids:
        raise ValueError("No overlap between embeds_dict and attr_dict keys.")

    labels = [attr_dict[item_id] for item_id in ids]
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    n_classes = len(le.classes_)
    if n_classes < 2:
        raise ValueError("PLS probe requires at least two classes.")

    X = np.stack([embeds_dict[item_id] for item_id in ids], axis=0).astype(np.float64)
    X_scaled = StandardScaler().fit_transform(X)
    Y = OneHotEncoder(sparse_output=False).fit_transform(y_int.reshape(-1, 1))

    max_components = min(len(ids) - 1, n_classes - 1)
    if max_components < 1:
        raise ValueError("PLS probe requires at least one valid component.")
    if n_components is not None:
        n_used = min(n_components, max_components)
    else:
        n_used = max_components

    pls = PLSRegression(n_components=n_used, scale=False)
    pls.fit(X_scaled, Y)
    coords = pls.transform(X_scaled)

    X_mnlogit = np.hstack([np.ones((len(ids), 1)), coords])
    model = MNLogit(y_int, X_mnlogit)
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)

    pvals_per_component = result.pvalues[1:, :]
    pvals_min = pvals_per_component.min(axis=1)
    reject, pvals_corrected, _, _ = multipletests(
        pvals_min,
        alpha=alpha,
        method="fdr_bh",
    )
    significant_components = [idx for idx, is_sig in enumerate(reject) if is_sig]

    coefs = result.params[1:, :]
    max_coef_labels = []
    for idx in range(n_used):
        class_idx = int(np.abs(coefs[idx]).argmax())
        label_idx = min(class_idx + 1, len(le.classes_) - 1)
        max_coef_labels.append(le.classes_[label_idx])

    results_table = pd.DataFrame(
        {
            "component": range(n_used),
            "p_value_raw": pvals_min,
            "p_value_corrected": pvals_corrected,
            "significant": reject,
            "max_coef_label": max_coef_labels,
        }
    )

    return {
        "pls": pls,
        "pls_coords": coords,
        "n_components_used": n_used,
        "label_encoder": le,
        "ids": ids,
        "results_table": results_table,
        "significant_components": significant_components,
        "x_loadings": pls.x_loadings_,
        "mnlogit_result": result,
        "random_state": random_state,
    }


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default))


def run_static_attribute_audit(
    embeds_dict: EmbeddingDict,
    attr_dict: AttributeDict,
    embedding_type: str,
    attribute: str,
    output_dir: Path | str,
    min_samples: int = 5,
    n_folds: int = 5,
    run_pca_components: bool = True,
    run_pls: bool = False,
    max_heatmap_items: int = 60,
    pca_variance_threshold: float = 0.95,
    random_state: int = 1337,
) -> dict[str, Any]:
    """
    Run the static Q1 audit suite for one embedding type and one attribute.
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / "figures"
    metrics_dir = output_dir / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    filtered_embeds, filtered_attrs, filter_meta = align_static_audit_data(
        embeds_dict,
        attr_dict,
        min_samples=min_samples,
    )
    n_items = filter_meta["n_items"]
    n_classes = filter_meta["n_classes"]
    if n_items == 0:
        raise ValueError(f"No items left after filtering {attribute!r}.")
    if n_classes < 2:
        raise ValueError(
            f"Static Q1 audit for {attribute!r} requires at least two classes "
            "after filtering."
        )

    result = {
        "embedding_type": embedding_type,
        "attribute": attribute,
        "n_items": n_items,
        "n_classes": n_classes,
        "chance": 1.0 / n_classes,
        "class_counts": filter_meta["class_counts"],
        "dropped_class_counts": filter_meta["dropped_class_counts"],
    }

    lr_res = logistic_probe(
        filtered_embeds,
        filtered_attrs,
        n_folds=n_folds,
        random_state=random_state,
    )
    result.update(
        {
            "lr_accuracy": lr_res["overall_accuracy"],
            "lr_cv_fold_accuracies": lr_res["cv_fold_accuracies"],
            "lr_n_folds_used": lr_res["n_folds_used"],
        }
    )
    (metrics_dir / f"classification_report_{embedding_type}_{attribute}.txt").write_text(
        lr_res["classification_report"]
    )
    _write_json(
        metrics_dir / f"logistic_probe_{embedding_type}_{attribute}.json",
        {
            "overall_accuracy": lr_res["overall_accuracy"],
            "cv_fold_accuracies": lr_res["cv_fold_accuracies"],
            "per_class_accuracy": lr_res["per_class_accuracy"],
            "class_names": lr_res["class_names"],
            "confusion_matrix": lr_res["confusion_matrix"],
            "n_folds_used": lr_res["n_folds_used"],
        },
    )

    sim_matrix, sorted_ids, sorted_attrs = cos_sim_mat(
        filtered_embeds,
        attribute_dict=filtered_attrs,
    )
    bc = block_coherence(sim_matrix, sorted_attrs)
    result["block_coherence"] = bc
    _write_json(
        metrics_dir / f"block_coherence_{embedding_type}_{attribute}.json",
        {"block_coherence": bc},
    )

    if n_items <= max_heatmap_items:
        fig = plot_heatmap(
            sim_matrix,
            sorted_ids,
            sorted_attrs,
            title=f"{embedding_type}: cosine similarity by {attribute}",
        )
        fig.savefig(
            figures_dir / f"heatmap_{embedding_type}_{attribute}.png",
            dpi=200,
            bbox_inches="tight",
        )
        import matplotlib.pyplot as plt

        plt.close(fig)

    if run_pca_components:
        try:
            pca_res = pca_component_probe(
                filtered_embeds,
                filtered_attrs,
                variance_threshold=pca_variance_threshold,
                random_state=random_state,
            )
            pca_res["results_table"].to_csv(
                metrics_dir / f"pca_components_{embedding_type}_{attribute}.csv",
                index=False,
            )
            result.update(
                {
                    "pca_n_significant": len(pca_res["significant_components"]),
                    "pca_n_components": pca_res["n_components_used"],
                }
            )
        except Exception as err:
            result["pca_error"] = str(err)
            result["pca_n_significant"] = None
            result["pca_n_components"] = None

    if run_pls:
        try:
            pls_res = pls_probe(
                filtered_embeds,
                filtered_attrs,
                random_state=random_state,
            )
            pls_res["results_table"].to_csv(
                metrics_dir / f"pls_components_{embedding_type}_{attribute}.csv",
                index=False,
            )
            result.update(
                {
                    "pls_n_significant": len(pls_res["significant_components"]),
                    "pls_n_components": pls_res["n_components_used"],
                }
            )
        except Exception as err:
            result["pls_error"] = str(err)
            result["pls_n_significant"] = None
            result["pls_n_components"] = None
    else:
        result["pls_n_significant"] = None
        result["pls_n_components"] = None

    _write_json(metrics_dir / f"summary_{embedding_type}_{attribute}.json", result)
    return result
