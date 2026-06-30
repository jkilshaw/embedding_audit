"""Static Q1 analysis functions for id -> embedding dictionaries."""

from collections import Counter
from collections.abc import Iterable
from typing import Any, Optional

import matplotlib
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from statsmodels.discrete.discrete_model import MNLogit
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
from matplotlib import patches as mpatches  # pylint: disable=wrong-import-position
from matplotlib import pyplot as plt  # pylint: disable=wrong-import-position

EmbeddingDict = dict[int, np.ndarray]
AttributeDict = dict[int, str]
RGBAColor = tuple[float, float, float, float]


def _sorted_ids(ids: Iterable[int]) -> list[int]:
    """
    Sort ids from embedding dictionary in order to keep results consistent
    """
    return sorted(ids)


def align_static_audit_data(
    embeds_dict: EmbeddingDict,
    attr_dict: AttributeDict,
    min_samples: int = 5,
) -> tuple[EmbeddingDict, AttributeDict, dict[str, Any]]:
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
    ids = _sorted_ids(item_id for item_id in embeds_dict if item_id in attr_dict and attr_dict[item_id] is not None)
    # use counter to get counts of each categorical metadata entry
    label_counts = Counter(attr_dict[item_id] for item_id in ids)
    # Discard attribute categories with fewer than min_samples examples.
    kept_labels = {label for label, count in label_counts.items() if count >= min_samples}
    # make a list of ids with low occurrence ids filtered out
    kept_ids = [item_id for item_id in ids if attr_dict[item_id] in kept_labels]

    # filter out the dictionaries using only ids from the kept_ids list
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
        "dropped_class_counts": {
            label: count
            for label, count in sorted(label_counts.items())
            if label not in kept_labels
        },
    }
    return filtered_embeds, filtered_attrs, metadata


def cos_sim_mat(
    embeds_dict: EmbeddingDict,
    attribute_dict: Optional[AttributeDict] = None,
) -> tuple[np.ndarray, list[int], Optional[list[str]]]:
    """
    Compute pairwise cosine similarity for id -> vector dictionaries.

    If attribute_dict is provided, ids are sorted by attribute label first so
    same-label examples form contiguous blocks in heatmaps.

    :param embeds_dict: embedding dictionary with key as team_id and value as the vector
    :param attribute_dict: attribute dictionary with key as team id and value as the string classification
    :return tuple[np.ndarray, list[int], Optional[list[str]]]
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


def plot_heatmap(
    sim_matrix: np.ndarray,
    ids: list[int],
    attributes: Optional[list[str]] = None,
    title: str = "Cosine Similarity Matrix",
    figsize: tuple[int, int] = (12, 10),
):
    """
    Plot a cosine similarity heatmap with optional attribute block boundaries.

    :param sim_matrix: cosine similarity matrix
    :param ids: list of ids in same order as the matrix
    :param attributes: list of attribute strings in same order as the matrix
    :param title: title of the plot
    :param figsize: figure size of the plot
    :return None
    """

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
        boundaries = [
            i + 0.5
            for i in range(len(attributes) - 1)
            if attributes[i] != attributes[i + 1]
        ]

        # for each of those boundaries, add a line between them to symbolize attribute blocks
        for boundary in boundaries:
            ax.axhline(boundary, color="black", linewidth=0.8, linestyle="--")
            ax.axvline(boundary, color="black", linewidth=0.8, linestyle="--")

        # list comprehension for color patches for each attribute
        handles = [
            mpatches.Patch(color=color_map[attr], label=attr)
            for attr in unique_attrs
        ]

        # add a legend to the plot with handle color patches as handles
        # place the legend relative to the axis box x=1.15 slightly to the right of the plot and y=1 at the top
        # in the upper left with fontsize 7 and opacity mostly opaque at .8
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.15, 1),
            loc="upper left",
            fontsize=7,
            framealpha=0.8,
        )

    # set the title and adjust spacing so everything fits better
    ax.set_title(title)
    fig.tight_layout()

    # return the figure back to the caller to decide what to do next
    return fig


def umap_embed(
    embeds_dict: EmbeddingDict,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 1337,
) -> dict[str, Any]:
    """
    Perform UMAP dimensionality reduction on static embeddings.

    :param embeds_dict: embedding dictionary with id keys and vector values
    :param n_components: number of UMAP dimensions
    :param n_neighbors: UMAP neighborhood size
    :param min_dist: minimum distance between low-dimensional points
    :param metric: distance metric for UMAP
    :param random_state: random seed
    :return dict: UMAP coordinates, ordered ids, and fitted UMAP model
    """

    # get the list of ids and build a feature matrix
    ids = _sorted_ids(embeds_dict)
    feat_matx = np.stack([embeds_dict[item_id] for item_id in ids], axis=0)

    try:
        import umap  # pylint: disable=import-outside-toplevel
    except ImportError as err:
        raise ImportError(
            "umap_embed requires the optional dependency 'umap-learn'. "
            "Install project requirements before running UMAP."
        ) from err

    # create model instance with parameterized values
    umap_model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )

    # generate a numpy array object of the umap_coordinates
    umap_coords = umap_model.fit_transform(feat_matx)

    # return the dictionary objects containing summary statistics
    return {
        "umap": umap_model,
        "umap_coords": umap_coords,
        "ids": ids,
        "n_components": n_components,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
    }


def plot_umap(
    umap_coords: np.ndarray,
    ids: list[int],
    components: tuple[int, int] = (0, 1),
    attribute_dict: Optional[AttributeDict] = None,
    title: str = "UMAP",
    figsize: tuple[int, int] = (9, 7),
    label_points: bool = False,
):
    """
    Plot UMAP coordinates as a scatter plot with optional attribute coloring.

    :param umap_coords: UMAP coordinate matrix
    :param ids: ordered ids matching UMAP coordinate rows
    :param components: two UMAP dimensions to plot ( defaults to 1 dimension on x and 2 dimensions on y)
    :param attribute_dict: optional id to attribute dictionary
    :param title: plot title
    :param figsize: figure size
    :param label_points: whether to annotate point ids
    :return: matplotlib Figure
    """
    # unpack x and y components
    comp_x, comp_y = components
    # get all coordinates for x and y directions from their columns and turn into 1D numpy arrays
    xs = umap_coords[:, comp_x]
    ys = umap_coords[:, comp_y]

    # create the matplotlib feature with its axis
    fig, ax = plt.subplots(figsize=figsize)

    # if labels were passed in, the function colors points by their respective labels
    if attribute_dict is not None:
        attributes = [attribute_dict.get(item_id) for item_id in ids]
        present_attrs = [attr for attr in attributes if attr is not None]
        color_map, unique_attrs = _make_attribute_colormap(present_attrs)

        # collect points that are missing labels
        grey_points = [(x_coord, y_coord) for x_coord, y_coord, attr in zip(xs, ys, attributes) if attr is None]
        # collect points that are colored
        colored_points = [(x_coord, y_coord, attr)for x_coord, y_coord, attr in zip(xs, ys, attributes)
                          if attr is not None]

        # plot grey points on the scatter plot
        if grey_points:
            grey_xs, grey_ys = zip(*grey_points)
            ax.scatter(
                grey_xs,
                grey_ys,
                c=["#cccccc"],  # color grey
                s=20,           # point size
                alpha=0.4,      # transparency
                linewidths=0,
                zorder=1,       # draw behind colored points
                label="_nolegend_",
            )

        # plot colored points with their corresponding color
        if colored_points:
            colored_xs, colored_ys, colored_attrs = zip(*colored_points)
            colors = [color_map[attr] for attr in colored_attrs]
            ax.scatter(
                colored_xs,
                colored_ys,
                c=colors,
                s=40,
                alpha=0.9,
                linewidths=0,
                zorder=2,
            )

        # create legend entries for each unique color/label type
        handles = [mpatches.Patch(color=color_map[attr], label=attr) for attr in unique_attrs]

        # Add the legend outside of the plot on the right
        ax.legend(
            handles=handles,
            bbox_to_anchor=(1.01, 1),
            loc="upper left",
            fontsize=8,
            framealpha=0.8,
        )
    # if there is no attribute dict, just plot everything with no color
    else:
        ax.scatter(xs, ys, s=40, alpha=0.8, linewidths=0)

    # if requested, add labels to points in one default color
    if label_points:
        for x_coord, y_coord, item_id in zip(xs, ys, ids):
            ax.annotate(
                str(item_id),
                (x_coord, y_coord),
                fontsize=5,
                alpha=0.7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    # label axes, set the title, remove tick marks, and adjust the layout so things fit
    ax.set_xlabel(f"UMAP dim {comp_x + 1}", fontsize=10)
    ax.set_ylabel(f"UMAP dim {comp_y + 1}", fontsize=10)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig


def block_coherence(sim_matrix: np.ndarray, attrs: list[str]) -> float:
    """
    Return mean same-label similarity minus mean cross-label similarity.

    :param sim_matrix: cosine similarity matrix
    :param attrs: list of attribute strings in same order as the matrix
    :return float: block coherence score for embedding and attribute type
    """

    # convert the attribute list into a numpy array for numpy broadcasting
    attrs_arr = np.array(attrs)
    # create a boolean matrix for if attribute labels are the same or different
    same_label = attrs_arr[:, None] == attrs_arr[None, :]
    # create a boolean mask matrix for the upper triangle and exclude the diagonol as diagonal values are self-similar
    # k = 1 mean 1 above the main diagonal
    upper = np.triu(np.ones(sim_matrix.shape, dtype=bool), k=1)
    # get matrix values that have the same label and are in the upper triangle in order to avoid duplicate counts
    within_mask = np.logical_and(same_label, upper)
    within = sim_matrix[within_mask]
    # get the different label similarities
    cross_mask = np.logical_and(np.logical_not(same_label), upper)
    cross = sim_matrix[cross_mask]

    # it cannot be run if there are no cross label or same label pairs
    if within.size == 0 or cross.size == 0:
        return float("nan")
    # mean(within similarities) - mean(cross similarities) block coherence calculation
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
    :param embeds_dict: embedding dictionary
    :param attr_dict: attribute dictionary
    :param n_folds: number of folds
    :param max_iter: maximum number of iterations
    :param random_state: random seed
    :return dict: probe results
    """
    # make a list of ids that overlap in both dicts
    ids = _sorted_ids(item_id for item_id in embeds_dict if item_id in attr_dict)
    if not ids:
        raise ValueError("No overlap between embeds_dict and attr_dict keys.")

    # get a list of the attributes in the same order as their ids
    labels = [attr_dict[item_id] for item_id in ids]
    # create a label encoder to map string class labels to integer class ids.
    le = LabelEncoder()
    # Fit the mapping on labels and transform labels into integer ids for sklearn.
    y = le.fit_transform(labels)

    # count how many examples are in each encoded class bin
    class_counts = np.bincount(y)
    # raise an error if it has less than 2 classes or bins
    if len(class_counts) < 2:
        raise ValueError("Logistic probe requires at least two classes.")

    # make sure we do not use more folds than the smallest class size
    effective_folds = min(n_folds, int(class_counts.min()))
    # need at least 2 folds for cross-validation
    if effective_folds < 2:
        raise ValueError("Logistic probe requires at least two samples per class.")

    # build the feature matrix for the linear probe through list comprehension of embedding vectors stacked
    # into rows
    feat_matx = np.stack([embeds_dict[item_id] for item_id in ids], axis=0).astype(np.float32)
    # create the logistics regression classifier
    #       max_iter uses the maximum optimization steps
    #       random_state makes shuffling/initial behavior reproducible
    #       multinomial uses true multiclass logistic regression methods
    #       l-bfgs is the optimization algorithm
    #       1.0 is the regularization strength
    clf = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        multi_class="multinomial",
        solver="lbfgs",
        C=1.0,
    )
    # Put scaling inside the estimator so cross-validation fits the scaler only on each training fold.
    probe_pipeline = make_pipeline(StandardScaler(), clf)

    # creates the cross-validation splitter, so each fold maintains normal class proportions
    # effective folds, shuffle, and random state all work with distribution while preserving for reproducibility
    cv = StratifiedKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=random_state,
    )

    # 1. Run to get fair prediction evaluation
    # Run cross-validation and give one prediction for every item, y_pred_cv is an array of predicted int labels.
    # The pipeline prevents feature preprocessing leakage by fitting StandardScaler inside each training fold.
    y_pred_cv = cross_val_predict(probe_pipeline, feat_matx, y, cv=cv)

    # 2. Run across each fold manually to see if there is variance between folds
    # create a list to store the accuracy for each fold
    fold_accuracies = []
    # loop through each cv split of [training ids, value ids]
    for train_idx, val_idx in cv.split(feat_matx, y):
        # create a fresh logistic regression model for this fold, so eahc fold starts clean
        clf_fold = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state,
            multi_class="multinomial",
            solver="lbfgs",
            C=1.0,
        )
        fold_pipeline = make_pipeline(StandardScaler(), clf_fold)

        # train the model only on the training rows
        fold_pipeline.fit(feat_matx[train_idx], y[train_idx])
        # use the trained model to predict the validation rows, compare to their labels, and append accuracy score
        fold_accuracies.append(
            accuracy_score(y[val_idx], fold_pipeline.predict(feat_matx[val_idx]))
        )

    # Fit one final pipeline on all data for downstream inspection.
    probe_pipeline.fit(feat_matx, y)
    fitted_scaler = probe_pipeline.named_steps["standardscaler"]
    fitted_clf = probe_pipeline.named_steps["logisticregression"]
    # convert the integer labels back to their strings for the actual and predicted labels
    y_true_labels = le.inverse_transform(y)
    y_pred_labels = le.inverse_transform(y_pred_cv)

    # create a dictionary and loops through each class type
    per_class_accuracy = {}
    for label in le.classes_:
        # creates a boolean mask if the class label matches the label we are evaluating
        mask = y_true_labels == label
        # label the classes as keys with the accuracy score as their value
        per_class_accuracy[label] = accuracy_score(
            y_true_labels[mask],
            y_pred_labels[mask],
        )

    return {
        "model": probe_pipeline,
        "scaler": fitted_scaler,
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
        "coef": fitted_clf.coef_,
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
    :param embeds_dict: embedding dictionary
    :param attr_dict: attribute dictionary
    :param n_components: number of PCA components
    :param variance_threshold: variance threshold for PCA components
    :param alpha: alpha parameter for PCA components significant cutoff for multiple-testing correction
    :param random_state: random seed
    :return dict: probe results
    """

    # keep ids that exist in both dicitonaries
    ids = _sorted_ids(item_id for item_id in embeds_dict if item_id in attr_dict)
    if not ids:
        raise ValueError("No overlap between embeds_dict and attr_dict keys.")

    # convert string labels to integer class Ids
    labels = [attr_dict[item_id] for item_id in ids]
    le = LabelEncoder()
    y = le.fit_transform(labels)
    # make sure there are at least 2 classes
    if len(le.classes_) < 2:
        raise ValueError("PCA component probe requires at least two classes.")

    # build the feature matrix for PCA with one embedding vector per row
    feat_matx = np.stack([embeds_dict[item_id] for item_id in ids], axis=0).astype(
        np.float64
    )
    # scale and transform
    feat_matx_scaled = StandardScaler().fit_transform(feat_matx)

    # find the maximum number of PCA components which cannot be more than the number of samples and dimensions
    max_components = min(len(ids), feat_matx_scaled.shape[1])
    # fit the pca with as many components as possible to inspect the full variance curve
    pca_full = PCA(n_components=max_components, random_state=random_state)
    pca_full.fit(feat_matx_scaled)
    # compute the cumulative explained variance
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    # if user provides n_components use those, if not calculate how many are needed
    if n_components is not None:
        n_used = min(n_components, max_components)
    else:
        n_used = int(np.searchsorted(cumvar, variance_threshold) + 1)
        n_used = min(n_used, max_components)

    # fits PCA again using only the selected number of components
    pca = PCA(n_components=n_used, random_state=random_state)
    # coords is the transformed data
    # components are now represented by PCA component coordinates instead of original embeddings dimensions
    pca_coords = pca.fit_transform(feat_matx_scaled)

    # Add an intercept column to the PCA coordinates. The first column will all be 1s
    mnlogit_feat_matx = np.hstack([np.ones((len(ids), 1)), pca_coords])
    # Fit multinomial logistic regression using PCA coordinates to predict the label
    model = MNLogit(y, mnlogit_feat_matx)
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)

    # get p-values for PCA components and drop the intercept row
    pvals_per_component = result.pvalues[1:, :]
    # for each PCA component, take the smallest p-value across class contrasts
    # asks if this component is significantly associated with at least one class contrast
    pvals_min = pvals_per_component.min(axis=1)
    # Apply Benjamini-Hochberg correction to see if they are significant after correction
    reject, pvals_corrected, _, _ = multipletests(
        pvals_min,
        alpha=alpha,
        method="fdr_bh",
    )
    # build a list of indexes if they are significant based on p-val from reject above
    significant_components = [idx for idx, is_sig in enumerate(reject) if is_sig]

    # get fitted coefficients for PCA components, excluding the intercept
    coefs = result.params[1:, :]
    max_coef_labels = []
    # for each component, find which class contrast has the largest absolute coefficient
    for idx in range(n_used):
        class_idx = int(np.abs(coefs[idx]).argmax())
        label_idx = min(class_idx + 1, len(le.classes_) - 1)
        max_coef_labels.append(le.classes_[label_idx])

    # build a dataframe table with one row per PCA component
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

    # return summary dictionary of all findings
    return {
        "pca": pca,
        "pca_coords": pca_coords,
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
    # keep only ids that exist in both dictionaries
    ids = _sorted_ids(item_id for item_id in embeds_dict if item_id in attr_dict)
    if not ids:
        raise ValueError("No overlap between embeds_dict and attr_dict keys.")

    # Build a label list in the same order as the ids
    labels = [attr_dict[item_id] for item_id in ids]
    # convert the labels to numerical representations
    le = LabelEncoder()
    y_int = le.fit_transform(labels)
    n_classes = len(le.classes_)
    # make sure that there are at least two classes
    if n_classes < 2:
        raise ValueError("PLS probe requires at least two classes.")

    # build an embedding feature matrix
    X = np.stack([embeds_dict[item_id] for item_id in ids], axis=0).astype(np.float64)
    # standardize the embedding directions and transform
    X_scaled = StandardScaler().fit_transform(X)
    # convert integer labels into one-hot label matrix
    Y = OneHotEncoder(sparse_output=False).fit_transform(y_int.reshape(-1, 1))

    # finds the maximum meaningful number of PLS components
    max_components = min(len(ids) - 1, n_classes - 1)

    # logic for the amount of components to use and defensive programming
    if max_components < 1:
        raise ValueError("PLS probe requires at least one valid component.")
    if n_components is not None:
        n_used = min(n_components, max_components)
    else:
        n_used = max_components

    # create the pls model to use and set scale to false as we manually scaled the feature vector before
    pls = PLSRegression(n_components=n_used, scale=False)
    # fot PLS to find components of X that best relate to Y
    pls.fit(X_scaled, Y)
    # get each item's coordinates/scores on the PLS components shape (N, n_used)
    coords = pls.transform(X_scaled)

    # add intercept columns for statsmodels
    X_mnlogit = np.hstack([np.ones((len(ids), 1)), coords])
    # fit multinomial logistic regression using PLS coordinates to predict labels
    # mainly to get p-values for components
    model = MNLogit(y_int, X_mnlogit)
    result = model.fit(method="lbfgs", maxiter=1000, disp=False)

    # get PLS components, excluding the intercept
    pvals_per_component = result.pvalues[1:, :]
    # for each component, take the smallest p-value across class contrasts
    pvals_min = pvals_per_component.min(axis=1)
    # Apply Benjamini-Hochberg correction to keep only confirmed singificant p-values
    reject, pvals_corrected, _, _ = multipletests(
        pvals_min,
        alpha=alpha,
        method="fdr_bh",
    )
    # collect significant component indexes
    significant_components = [i for i, is_sig in enumerate(reject) if is_sig]

    # get component coefficients from MNLogit, excluding the intercept
    coefs = result.params[1:, :]
    max_coef_labels = []
    # for each PLS component, find which class contrast has the largest absolute coefficient
    for idx in range(n_used):
        class_idx = int(np.abs(coefs[idx]).argmax())
        label_idx = min(class_idx + 1, len(le.classes_) - 1)
        max_coef_labels.append(le.classes_[label_idx])

    # build one row per PLS component in this dataframe
    results_table = pd.DataFrame(
        {
            "component": range(n_used),
            "p_value_raw": pvals_min,
            "p_value_corrected": pvals_corrected,
            "significant": reject,
            "max_coef_label": max_coef_labels,
        }
    )

    # return dictionary of summary statistics
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
