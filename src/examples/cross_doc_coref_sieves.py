import logging
from typing import List

from src import LIBRARY_ROOT
from src.cdc_resources.relations.referent_dict_relation_extraction import ReferentDictRelationExtraction
from src.cdc_resources.relations.relation_types_enums import RelationType
from src.cdc_resources.relations.wikipedia_relation_extraction import WikipediaRelationExtraction
from src.cdc_resources.relations.word_embedding_relation_extraction import WordEmbeddingRelationExtraction
from src.cross_doc_sieves import run_event_coref, run_entity_coref
from src.obj.cluster import Clusters
from src.obj.sieves_config import EventSievesConfiguration, EntitySievesConfiguration
from src.obj.sieves_resource import SievesResources
from src.obj.topics import Topics
from src.sieves_container_init import SievesContainerInitialization


def run_example(cdc_settings):
    event_mentions_topics = Topics()
    event_mentions_topics.create_from_file(str(LIBRARY_ROOT / 'datasets' / 'ecb'
                                               / 'ecb_all_event_mentions.json'))

    event_clusters = None
    if cdc_settings.event_config.run_evaluation:
        logger.info('Running event coreference resolution')
        event_clusters = run_event_coref(event_mentions_topics, cdc_settings)

    entity_mentions_topics = Topics()
    entity_mentions_topics.create_from_file(str(LIBRARY_ROOT / 'datasets' / 'ecb'
                                                / 'ecb_all_entity_mentions.json'))
    entity_clusters = None
    if cdc_settings.entity_config.run_evaluation:
        logger.info('Running entity coreference resolution')
        entity_clusters = run_entity_coref(entity_mentions_topics, cdc_settings)

    return event_clusters, entity_clusters


def load_modules(cdc_resources):
    models = list()
    models.append(WikipediaRelationExtraction(cdc_resources.wiki_search_method,
                                              wiki_file=cdc_resources.wiki_folder,
                                              host=cdc_resources.elastic_host,
                                              port=cdc_resources.elastic_port,
                                              index=cdc_resources.elastic_index))
    models.append(WordEmbeddingRelationExtraction(cdc_resources.embed_search_method,
                                                  glove_file=cdc_resources.glove_file,
                                                  elmo_file=cdc_resources.elmo_file,
                                                  cos_accepted_dist=0.75))
    models.append(ReferentDictRelationExtraction(cdc_resources.referent_dict_method,
                                                 cdc_resources.referent_dict_file))
    return models


def create_example_settings():
    event_config = EventSievesConfiguration()
    event_config.sieves_order = [
        (RelationType.SAME_HEAD_LEMMA, 1.0),
        (RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
        (RelationType.WORD_EMBEDDING_MATCH, 0.7),
    ]

    entity_config = EntitySievesConfiguration()
    entity_config.sieves_order = [
        (RelationType.SAME_HEAD_LEMMA, 1.0),
        (RelationType.WIKIPEDIA_REDIRECT_LINK, 0.1),
        (RelationType.WIKIPEDIA_DISAMBIGUATION, 0.1),
        (RelationType.WORD_EMBEDDING_MATCH, 0.7),
        (RelationType.REFERENT_DICT, 0.5)
    ]

    # CDCResources hold default attribute values that might need to be change,
    # (using the defaults values in this example), use to configure attributes
    # such as resources files location, output directory, resources init methods and other.
    # check in class and see if any attributes require change in your set-up
    resource_location = SievesResources()
    return SievesContainerInitialization(event_config, entity_config,
                                         load_modules(resource_location))


def print_results(clusters: List[Clusters], type: str):
    print('-=' + type + ' Clusters=-')
    for topic_cluster in clusters:
        print('\n\tTopic=' + topic_cluster.topic_id)
        for cluster in topic_cluster.clusters_list:
            cluster_mentions = list()
            for mention in cluster.mentions:
                mentions_dict = dict()
                mentions_dict['id'] = mention.mention_id
                mentions_dict['text'] = mention.tokens_str
                cluster_mentions.append(mentions_dict)

            print('\t\tCluster(' + str(cluster.coref_chain) + ') Mentions='
                  + str(cluster_mentions))


def run_cdc_pipeline():
    cdc_settings = create_example_settings()
    event_clusters, entity_clusters = run_example(cdc_settings)

    print('-=Cross Document Coref Results=-')
    print_results(event_clusters, 'Event')
    print('################################')
    print_results(entity_clusters, 'Entity')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    run_cdc_pipeline()
