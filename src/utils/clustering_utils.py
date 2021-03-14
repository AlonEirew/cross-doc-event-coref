import logging
import time
from itertools import product

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering

from dataobjs.cluster import Cluster, Clusters
from dataobjs.topics import Topic

logger = logging.getLogger(__name__)


MAX_ALLOWED_BATCH_SIZE = 20000


def agglomerative_clustering(inference_model, topic: [Topic], average_link_thresh):
    clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed', linkage='average',
                                         distance_threshold=average_link_thresh)
    start = time.time()
    all_pairs = list(product(topic.mentions, repeat=2))
    pairs_chunks = [all_pairs]
    if len(all_pairs) > MAX_ALLOWED_BATCH_SIZE:
        pairs_chunks = [all_pairs[i:i + MAX_ALLOWED_BATCH_SIZE] for i in range(0, len(all_pairs), MAX_ALLOWED_BATCH_SIZE)]

    predictions = np.empty(0)
    with torch.no_grad():
        for chunk in pairs_chunks:
            chunk_predictions, _ = inference_model.predict(chunk, bs=len(chunk))
            predictions = np.append(predictions, chunk_predictions.detach().cpu().numpy())

    predictions = 1 - predictions
    pred_matrix = predictions.reshape(len(topic.mentions), len(topic.mentions))
    fit = clustering.fit(pred_matrix)
    for i in range(len(topic.mentions)):
        topic.mentions[i].predicted_coref_chain = fit.labels_[i] + Clusters.get_cluster_coref_chain()

    total_merged_clusters = max(fit.labels_)
    Clusters.inc_cluster_coref_chain(total_merged_clusters + 1)

    end = time.time()
    took = end - start
    logger.info('Total of %d clusters merged using method: %s, took: %.4f sec',
                total_merged_clusters, "Agglomerative Clustering", took)

    return topic.mentions
