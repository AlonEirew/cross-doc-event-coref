import logging
import time
from itertools import product

import torch
from enum import Enum
from sklearn.cluster import AgglomerativeClustering

from src.dataobjs.cluster import Cluster, Clusters
from src.dataobjs.topics import Topic
from src.coref_system.relation_extraction import RelationExtraction
from src.pairwize_model.model import PairWiseModel

logger = logging.getLogger(__name__)


class ClusteringType(Enum):
    AgglomerativeClustering = 0
    NaiveClustering = 1


def agglomerative_clustering(inference_model, topic: [Topic], average_link_thresh):
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         distance_threshold=average_link_thresh)
    start = time.time()
    all_pairs = list(product(topic.mentions, repeat=2))

    with torch.no_grad():
        predictions, _ = inference_model.predict(all_pairs, bs=len(all_pairs))

    predictions = 1 - predictions.detach().cpu().numpy()
    pred_matrix = predictions.reshape(len(topic.mentions), len(topic.mentions))
    fit = clustering.fit(pred_matrix)
    for i in range(len(topic.mentions)):
        topic.mentions[i].predicted_coref_chain = fit.labels_[i] + Clusters.get_cluster_coref_chain()

    total_merged_clusters = max(fit.labels_)
    Clusters.inc_cluster_coref_chain(total_merged_clusters)

    end = time.time()
    took = end - start
    logger.info('Total of %d clusters merged using method: %s, took: %.4f sec',
                total_merged_clusters, "Agglomerative Clustering", took)

    return topic.mentions


def naive_clustering(extractor: RelationExtraction, topic: [Topic], average_link_thresh):
    # pylint: disable=too-many-nested-blocks
    logger.info('loading topic %s, total mentions: %d', topic.topic_id, len(topic.mentions))
    clusters = Clusters(topic.topic_id, topic.mentions)

    start = time.time()
    clusters_changed = True
    merge_count = 0
    while clusters_changed:
        clusters_changed = False
        clusters_size = len(clusters.clusters_list)
        for i in range(0, clusters_size):
            cluster_i = clusters.clusters_list[i]
            if cluster_i.merged:
                continue

            for j in range(i + 1, clusters_size):
                cluster_j = clusters.clusters_list[j]
                if cluster_j.merged:
                    continue

                if cluster_i is not cluster_j:
                    criterion = test_cluster_merge(extractor, cluster_i, cluster_j, average_link_thresh)
                    if criterion:
                        merge_count += 1
                        clusters_changed = True
                        cluster_i.merge_clusters(cluster_j)
                        cluster_j.merged = True

        if clusters_changed:
            clusters.clean_clusters()

        end = time.time()
        took = end - start
        logger.info('Total of %d clusters merged using Naive Clustering method, relation: %s, took: %.4f sec',
                    merge_count, str(extractor.get_supported_relation()), took)

    return topic.mentions


def test_cluster_merge(extractor: RelationExtraction, cluster_i: Cluster, cluster_j: Cluster, average_link_thresh) -> bool:
    """
    Args:
        :param extractor:
        :param cluster_i:
        :param cluster_j:
        :param average_link_thresh:

    Returns:
        bool -> indicating whether to merge clusters (True) or not (False)
    """
    expected_relations = extractor.get_supported_relation()
    matches = 0
    for mention_i in cluster_i.mentions:
        for mention_j in cluster_j.mentions:
            match_result = extractor.solve(mention_i, mention_j)
            if match_result == expected_relations:
                matches += 1

    possible_pairs_len = float(len(cluster_i.mentions) * len(cluster_j.mentions))
    matches_rate = matches / possible_pairs_len

    result = False
    if matches_rate >= average_link_thresh:
        result = True

    return result
