import logging
import time

from sklearn.cluster import AgglomerativeClustering
from src.dataobjs.cluster import Clusters

logger = logging.getLogger(__name__)


def agglomerative_clustering(pred_matrix, topic, average_link_thresh):
    start = time.time()
    total_merged_clusters = run_clustering(pred_matrix, topic, average_link_thresh)

    end = time.time()
    took = end - start
    logger.info('Total of %d clusters merged using method: %s, took: %.4f sec',
                total_merged_clusters, "Agglomerative Clustering", took)

    return topic.mentions


def run_clustering(pred_matrix, topic, average_link_thresh):
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         distance_threshold=average_link_thresh)
    fit = clustering.fit(pred_matrix)
    for i in range(len(topic.mentions)):
        topic.mentions[i].predicted_coref_chain = fit.labels_[i] + Clusters.get_cluster_coref_chain()
    total_merged_clusters = max(fit.labels_)
    Clusters.inc_cluster_coref_chain(total_merged_clusters + 1)
    return total_merged_clusters
