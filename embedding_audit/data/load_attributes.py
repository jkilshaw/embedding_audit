import json
from pathlib import Path
from typing import Any

from embedding_audit.data.load_embeddings import (
    _parse_id_from_token as parse_id_from_token,
    load_meta,
)

AttributeDict = dict[int, str]


def load_index(index_path: str) -> dict[str, Any]:
    """
    Load the processed match index JSON
    It contains mathc information from statsbomb data

    :param index_path: path to the JSON file
    :return dict[str, Any] of the Json data as a Python object
    """
    with open(index_path, "r") as f:
        return json.load(f)


def make_league_dict(team_ids: list[int], matches: list[dict[str, Any]]) -> AttributeDict:
    """
    Build {team_id: league} using the first match where each team appears.
    Basically loop through each match dict in the matches list of dicts, get the statsbomb ID numbers,
    check if this number is one we are looking for from the set we made from the team_ids and check
    if the number has been inserted in our team_to_league dict, and then if yes and no, it is added to the
    team_to_league dict where the league is the value and the ID is the key

    This mirrors the existing helper in tools.py without importing that module's
    full analysis dependency stack.
    :param team_ids: list of team_ids to check for in the matches
    :param matches: list of matches to check for in the matches
    :return dict[str, Any] of the Json data as a Python object. Key is the teamID and value is the league
    """

    # convert the list of teams to a set for faster look up
    team_set = set(team_ids)
    # create the empty dict for the teamID, league pairs
    team_to_league: AttributeDict = {}

    # loop through the list of dicts of each match and get the team IDs involved
    # then assign these to their league. int is used for defensive programming
    for match in matches:
        league = str(match["league"])
        for team_id in (int(match["home_team_id"]), int(match["away_team_id"])):
            if team_id in team_set and team_id not in team_to_league:
                team_to_league[team_id] = league

    # alert the user if some team_set ids are not found in the team_to_league metadata
    missing = team_set - team_to_league.keys()
    if missing:
        print(f"[WARN] {len(missing)} team(s) not found in any match: {missing}")

    return team_to_league


def league_to_sex(league: str) -> str:
    """
    Match the legacy audit rule from tools.py.
    Check if league is of a lnown womens league and label sex as Women
    Put all others as Men
    :param league: legacy audit rule from tools.py of known female league's
    :return str: string of men or women label

    """
    womens = {
        "England - FA Women's Super League",
        "Germany - Frauen Bundesliga",
        "Italy - Serie A Women",
        "Spain - Liga F",
        "United States of America - NWSL",
    }
    return "women" if league in womens else "men"


def load_team_attributes(meta_path: str, index_path: str) -> dict[str, AttributeDict]:
    """
    Return Q1 team attributes keyed by real StatsBomb team id.

    Static team embeddings have one vector per team, so this only returns labels
    that naturally map one team id to one value.
    """
    meta = load_meta(meta_path)
    data_dict = load_index(index_path)

    team_token_to_id = meta["teams"]["token_to_id"]
    matches = data_dict["matches"]

    all_team_ids = []
    for token in team_token_to_id:
        team_id = parse_id_from_token(str(token))
        if team_id is not None:
            all_team_ids.append(team_id)

    team_to_league = make_league_dict(all_team_ids, matches)
    team_to_sex = {
        team_id: league_to_sex(league)
        for team_id, league in team_to_league.items()
    }

    return {
        "league": team_to_league,
        "sex": team_to_sex,
    }


def load_q1_attributes(
    embedding_type: str,
    meta_path: str | Path,
    index_path: str | Path,
    team_season_meta_path: str | Path | None = None,
    player_to_team_path: str | Path | None = None,
    player_to_position_path: str | Path | None = None,
    position_categories_path: str | Path | None = None,
    team_season_weight=None,
    team_embedding_dict=None,
) -> dict[str, AttributeDict]:
    """
    Load attribute dictionaries for static Q1 embedding audits.

    Currently implemented:
      - team: league, sex
    """
    if embedding_type == "team":
        return load_team_attributes(meta_path, index_path)

    raise NotImplementedError(f"Q1 attributes are not implemented for {embedding_type!r}.")
