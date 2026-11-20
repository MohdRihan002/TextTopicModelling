
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF


def perform_topic_modeling(
    documents,
    num_topics=3,
    algorithm="LDA",
    num_keywords=10
):

    # Select vectorizer according to algorithm
    if algorithm == "LDA":
        vectorizer = CountVectorizer(
            stop_words="english",
            max_features=1000
        )
    else:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=1000
        )

    document_matrix = vectorizer.fit_transform(documents)

    if document_matrix.shape[1] == 0:
        raise ValueError(
            "Not enough meaningful words. Please add more text."
        )

    actual_topics = min(
        num_topics,
        len(documents),
        document_matrix.shape[1]
    )

    # Select topic modelling algorithm
    if algorithm == "LDA":

        model = LatentDirichletAllocation(
            n_components=actual_topics,
            random_state=42,
            max_iter=10
        )

    elif algorithm == "NMF":

        model = NMF(
            n_components=actual_topics,
            random_state=42,
            max_iter=200
        )

    else:
        raise ValueError("Please select LDA or NMF.")

    model.fit(document_matrix)

    words = vectorizer.get_feature_names_out()

    topic_results = []

    for topic_index, topic in enumerate(model.components_):

        top_words = [
            words[i]
            for i in topic.argsort()[-num_keywords:][::-1]
        ]

        topic_results.append({
            "Topic": topic_index + 1,
            "Keywords": top_words
        })

    document_topics = model.transform(document_matrix)

    # Normalize NMF scores for topic distribution
    if algorithm == "NMF":
        row_sums = document_topics.sum(axis=1, keepdims=True)

        document_topics = np.divide(
            document_topics,
            row_sums,
            out=np.zeros_like(document_topics),
            where=row_sums != 0
        )

    return topic_results, document_topics
