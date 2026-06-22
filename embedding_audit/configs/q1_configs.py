"""
Python configuration file for initial embedding label audits to be tested

Will contain dictionary structure of embedding type and matched labeling types to test
"""

# Data structure of embedding type and matched labeling for analysis
Q1_AUDITS = {
    "team": [
        "league",
        "sex",
        "season",
    ],

    "team_season": [
        "team",
        "league",
        "season",
        "sex",
    ],

    "combined": [
        "league",
        "season",
        "sex",
    ],

    "player": [
        "team",
        "league",
        "position_group",
        "position",
        "sex",
    ],
}


def get_q1_config(embedding_type: str) -> tuple[str, list[str]]:
    """

    :param embedding_type: which embedding type to test
    :return: tuple of embedding type and matched labeling types to test
    """

    if embedding_type not in Q1_AUDITS:
        raise ValueError()
    return embedding_type, Q1_AUDITS[embedding_type]