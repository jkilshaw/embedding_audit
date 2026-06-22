"""
Pipeline to run analysis for question 1:
Which axis do embeddings separate data on?
AKA:
What exactly do these initial embeddings encode for?
"""

import sys

import numpy as np

from embedding_audit.configs.q1_configs import get_q1_config


def load_q1_data_embedding(embedding_type: str) -> dict[int, np.ndarray]:
    """
    Return a dictionary representing the embedding matrix of a given choice
    Keys will be an ID corresponding to a value vector

    :param embedding_type: type of embedding to gather data from
    :return: dictionary of embedding id as key and its vector as value
    """



def run_q1(embedding_name: str) -> None:
    """
    Run the pipeline to answer question 1 for a specific embedding type

    :param embedding_name: which embedding we are running on
    """

    # unpack embedding type, and its attributes or labels to test against
    embedding_type, attributes = get_q1_config(embedding_name)

    print(f"Embedding type: {embedding_type}")
    print(f"Attributes to test: {attributes}")

    # run analysis for each attribute for the embedding type
    for attribute in attributes:
        print(f"\nRunning {embedding_type} embedding audit for label: {attribute}")
        # TODO:
        # run_pca(...)
        # run_cosine_similarity(...)
        # run_clustered_cosine(...)
        # run_block_coherence(...)
        # run_logistic_probe(...)


if __name__ == "__main__":
    embedding = sys.argv[1] if len(sys.argv) > 1 else "team"
    run_q1(embedding)
