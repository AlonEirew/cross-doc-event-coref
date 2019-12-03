import logging
from typing import List

from src.dataobjs.cluster import Clusters
from src.dataobjs.topics import Topics
from src.dt_system.run_sieve_system import get_run_system
from src.dt_system.sieves_container_init import SievesContainerInitialization

logger = logging.getLogger(__name__)


def run_event_coref(topics: Topics, resources: SievesContainerInitialization) -> List[Clusters]:
    """
    Running Cross Document Coref on event mentions
    Args:
        topics   : The Topics (with mentions) to evaluate
        resources: resources for running the evaluation

    Returns:
        Clusters: List of clusters and mentions with predicted cross doc coref within each topic
    """

    return _run_coref(topics, resources, 'event')


def run_entity_coref(topics: Topics, resources: SievesContainerInitialization) -> List[Clusters]:
    """
    Running Cross Document Coref on Entity mentions
    Args:
        topics   : The Topics (with mentions) to evaluate
        resources: (SievesContainerInitialization) resources for running the evaluation

    Returns:
        Clusters: List of topics and mentions with predicted cross doc coref within each topic
    """
    return _run_coref(topics, resources, 'entity')


def _run_coref(topics: Topics, resources: SievesContainerInitialization,
               eval_type: str) -> List[Clusters]:
    """
    Running Cross Document Coref on Entity mentions
    Args:
        resources: (SievesContainerInitialization) resources for running the evaluation
        topics   : The Topics (with mentions) to evaluate

    Returns:
        Clusters: List of topics and mentions with predicted cross doc coref within each topic
    """
    clusters_list = list()
    for topic in topics.topics_list:
        sieves_list = get_run_system(topic, resources, eval_type)
        clusters = sieves_list.run_deterministic()
        clusters.set_coref_chain_to_mentions()
        clusters_list.append(clusters)

    return clusters_list
