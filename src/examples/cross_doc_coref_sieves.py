import logging
from typing import List

from src import LIBRARY_ROOT
from src.cdc_resources.relations.computed_relation_extraction import ComputedRelationExtraction
from src.cdc_resources.relations.referent_dict_relation_extraction import ReferentDictRelationExtraction
from src.cdc_resources.relations.relation_types_enums import RelationType, WikipediaSearchMethod
from src.cdc_resources.relations.wikipedia_relation_extraction import WikipediaRelationExtraction
from src.cdc_resources.relations.word_embedding_relation_extraction import WordEmbeddingRelationExtraction
from src.cross_doc_sieves import run_event_coref, run_entity_coref
from src.obj.cluster import Clusters
from src.obj.sieves_config import EventSievesConfiguration, EntitySievesConfiguration
from src.obj.sieves_resource import SievesResources
from src.obj.topics import Topics
from src.sieves_container_init import SievesContainerInitialization
from src.utils.io_utils import write_coref_scorer_results


def run_example(cdc_settings):
    event_mentions_topics = Topics()
    event_mentions_topics.create_from_file(str(LIBRARY_ROOT / 'resources' / 'ecb' / 'gold_processed' / 'kian'
                                               / 'ECB_Test_Event_gold_mentions.json'))

    event_clusters = None
    if cdc_settings.event_config.run_evaluation:
        logger.info('Running event coreference resolution')
        event_clusters = run_event_coref(event_mentions_topics, cdc_settings)

    entity_mentions_topics = Topics()
    entity_mentions_topics.create_from_file(str(LIBRARY_ROOT / 'resources' / 'ecb' / 'gold_processed' / 'kian'
                                                / 'ECB_Test_Entity_gold_mentions.json'))
    entity_clusters = None
    if cdc_settings.entity_config.run_evaluation:
        logger.info('Running entity coreference resolution')
        entity_clusters = run_entity_coref(entity_mentions_topics, cdc_settings)

    return event_clusters, entity_clusters


def load_modules(cdc_resources):
    models = list()
    # models.append(WikipediaRelationExtraction(cdc_resources.wiki_search_method,
    #                                           wiki_file=cdc_resources.wiki_folder,
    #                                           host=cdc_resources.elastic_host,
    #                                           port=cdc_resources.elastic_port,
    #                                           index=cdc_resources.elastic_index))
    # models.append(WordEmbeddingRelationExtraction(cdc_resources.embed_search_method,
    #                                               glove_file=cdc_resources.glove_file,
    #                                               elmo_file=cdc_resources.elmo_file,
    #                                               cos_accepted_dist=0.75))
    # models.append(ReferentDictRelationExtraction(cdc_resources.referent_dict_method,
    #                                              cdc_resources.referent_dict_file))
    models.append(ComputedRelationExtraction())
    return models


def create_example_settings():
    event_config = EventSievesConfiguration()
    event_config.sieves_order = [
        (RelationType.SAME_HEAD_LEMMA, 1.0)
    ]

    entity_config = EntitySievesConfiguration()
    entity_config.sieves_order = [
        (RelationType.SAME_HEAD_LEMMA, 1.0)
    ]

    # CDCResources hold default attribute values that might need to be change,
    # (using the defaults values in this example), use to configure attributes
    # such as resources files location, output directory, resources init methods and other.
    # check in class and see if any attributes require change in your set-up
    resource_location = SievesResources()
    resource_location.wiki_search_method = WikipediaSearchMethod.ELASTIC
    return SievesContainerInitialization(event_config, entity_config,
                                         load_modules(resource_location))


def run_cdc_pipeline(print_method):
    cdc_settings = create_example_settings()
    event_clusters, entity_clusters = run_example(cdc_settings)

    print('-=Cross Document Coref Results=-')
    print_method(event_clusters, 'Event')
    print('################################')
    print_method(entity_clusters, 'Entity')


def print_scorer_results(all_clusters, eval_type):
    if eval_type == 'Event':
        out_file = str(LIBRARY_ROOT / 'output' / 'event_scorer_results.txt')
    else:
        out_file = str(LIBRARY_ROOT / 'output' / 'entity_scorer_results.txt')

    all_mentions = Clusters.from_clusters_to_mentions_list(all_clusters)
    write_coref_scorer_results(all_mentions, out_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # print_method = print_cluster_results
    print_methods = print_scorer_results

    run_cdc_pipeline(print_methods)
