from pathlib import Path
import pickle
from typing import Any
import numpy as np
import torch


EmbeddingDict = dict[int, np.ndarray]


def load_checkpoint(checkpoint_path: str) -> dict[str, Any]:
    """
    Load a model checkpoint from disk.

    Q1 uses static embedding tables stored under ckpt["model"]
    some of these are: "team_embed.weight", "team_season_embed.weight", and "player_embed.weight".
    :param checkpoint_path: Path to checkpoint file
    :return: Dictionary of embeddings from that file
    """
    return torch.load(checkpoint_path, map_location="cpu", weights_only=False)


def load_meta(meta_path: str) -> dict[str, Any]:
    """
    Load a pickle metadata file

    :param meta_path: Path to metadata file
    :return: Dictionary of metadata from that file
    """
    with open(meta_path, "rb") as f:
        return pickle.load(f)


def _get_weight(ckpt: dict[str, Any], weight_key: str) -> torch.Tensor:
    """
    Return one static embedding table from ckpt['model'] with a clear error.

    :param ckpt: Dictionary of embeddings from that file
    :param weight_key: Specific embedding type to get weight matrix from
    :return: Static embedding matrix
    """

    try:
        return ckpt["model"][weight_key]
    # if the key is not found, tell user and store that as the new error description
    except KeyError as err:
        raise KeyError(
            f"Checkpoint does not contain {weight_key!r}. "
        ) from err


def _tensor_to_numpy(weight: torch.Tensor) -> np.ndarray:
    """
    Convert a full embedding matrix into a detached CPU numpy matrix so it can be used.

    :param weight: Static embedding matrix
    :return: Detached CPU numpy matrix
    """
    return weight.detach().cpu().numpy()


def _parse_id_from_token(token: str) -> int | None:
    """
    Parse the integer entity id from tokens like '749:tottenham_hotspur_women'
    or from a plain integer-like token.
    Split on the : and take the first value

    :param token: Token to parse
    :return: Integer entity id
    """
    raw_id = token.split(":", 1)[0]
    try:
        return int(raw_id)
    except ValueError:
        return None


def load_team_embeddings(
    checkpoint_path: str | Path,
    meta_path: str | Path,
) -> EmbeddingDict:
    """
    Return static team embeddings as {team_id: embedding_vector}.

    The team id is parsed from meta["teams"]["token_to_id"] tokens such as
    "749:tottenham_hotspur_women". Special/non-entity tokens are skipped.
    """

    # Run function to get the .pt checkpoint loaded into memory as a dictionary object
    ckpt = load_checkpoint(checkpoint_path)

    # Run function to load and convert the metadata pickle file to a dictionary object
    meta = load_meta(meta_path)

    # get the static embedding matrix from its key from the checkpoint dictionary
    # then convert it to a numpy matrix
    team_weight = _tensor_to_numpy(_get_weight(ckpt, "team_embed.weight"))

    # get the token to id value component of the metadata dict of dict of dict object
    team_token_to_id = meta["teams"]["token_to_id"]

    # loop through each value of the token to id dict gathered above and parse the key column
    # use this key's first value, if there, to be the key for the team_embeds dictionary, and match it
    # with the same order of index that the embedding dictionary had
    team_embeds: EmbeddingDict = {}
    for token, index in team_token_to_id.items():
        team_id = _parse_id_from_token(token)
        if team_id is None:
            continue

        team_embeds[team_id] = team_weight[index].copy()

    return team_embeds
