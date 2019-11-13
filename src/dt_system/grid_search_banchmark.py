import logging

import numpy

from src import LIBRARY_ROOT
from src.dt_system.cross_doc_sieves import _run_coref
from src.obj.sieves_config import EventSievesConfiguration, EntitySievesConfiguration
from src.obj.sieves_resource import SievesResources
from src.obj.topics import Topics
from src.dt_system.sieves_container_init import SievesContainerInitialization
from src.utils.bcubed_scorer import bcubed

logger = logging.getLogger(__name__)


def run_banchmark():
    weights = numpy.arange(0.1, 1.1, 0.1)

    event_sieves_list = [
        # RelationType.OTHER,

    ]

    entity_sieves_list = [

    ]

    entity_topics, event_topics = load_mentions()

    best_ordered_event, event_results = calc_single_sieve_baselines(event_sieves_list, event_topics, weights, 'event')
    best_ordered_entity, entity_results = calc_single_sieve_baselines(entity_sieves_list, entity_topics, weights, 'entity')

    best_ordered_event.sort(key=lambda x: x[0])
    best_ordered_entity.sort(key=lambda x: x[0])

    final_event, event_record = find_best_sieves_order(best_ordered_event, event_topics, 'event', 1)
    final_entity, entity_record = find_best_sieves_order(best_ordered_entity, entity_topics, 'entity', 1)

    write_report(best_ordered_event, event_results, event_record, final_event, event_report_out)
    write_report(best_ordered_entity, entity_results, entity_record, final_entity, entity_report_out)


def find_best_sieves_order(best_ordered_orig, topics, conf_type, num_of_passes):
    base_line = 0.0
    record_results = list()
    best_ordered = best_ordered_orig.copy()
    best_sieves_weight_order = []
    for j in range(num_of_passes):
        for i in range(len(best_ordered) - 1, -1, -1):
            sieve_order = best_ordered[i]
            add_on_best = best_sieves_weight_order.copy()
            rel_type = sieve_order[1]
            weight = sieve_order[2]
            add_on_best.append((rel_type, weight))
            reset_coref_predicted_chain(topics)

            event_config, entity_config = create_sieves_order(add_on_best, conf_type)
            cdc_settings = create_example_settings(event_config, entity_config)
            recall, precision, b3_f1 = run_benchmark_on_config(topics, cdc_settings, conf_type)

            if conf_type == 'event':
                record_results.append((recall, precision, b3_f1, event_config.sieves_order))
            elif conf_type == 'entity':
                record_results.append((recall, precision, b3_f1, entity_config.sieves_order))

            if base_line < b3_f1:
                base_line = b3_f1
                best_sieves_weight_order.append((rel_type, weight))
                del best_ordered[i]

    return best_sieves_weight_order, record_results


def calc_single_sieve_baselines(sieves_list, topics, weights, conf_type):
    results = list()
    best_ordered = list()
    for rel_type in sieves_list:
        best_prec = 0
        best_weight = 0
        for weight in weights:
            reset_coref_predicted_chain(topics)
            rel_types_weights = [(rel_type, weight)]
            event_config, entity_config = create_sieves_order(rel_types_weights, conf_type)
            cdc_settings = create_example_settings(event_config, entity_config)

            recall, precision, b3_f1 = run_benchmark_on_config(topics, cdc_settings, conf_type)
            if conf_type == 'event':
                results.append((recall, precision, b3_f1, event_config.sieves_order))
            elif conf_type == 'entity':
                results.append((recall, precision, b3_f1, entity_config.sieves_order))

            # prefer higher weights in case equal
            if best_prec <= b3_f1:
                best_prec = b3_f1
                best_weight = weight

        best_ordered.append((best_prec, rel_type, best_weight))

    return best_ordered, results


def write_report(best_ordered, results, records, final, ofile):
    with open(ofile, 'w') as output_file:
        output_file.write('#####################\n')
        output_file.write('REPORT ' + ofile + '\n')
        output_file.write('CALCULATIONS:\n')
        output_file.write('\n'.join([str(res) for res in results]))
        output_file.write('\n\nBEST SIEVES ORDERED (BY PRECISION):\n')
        output_file.write('\n'.join([str(best_result) for best_result in best_ordered]))
        output_file.write('\n\n')
        output_file.write('FINAL SELECTED SIEVES RECORD (BY F1)\n')
        output_file.write('\n'.join([str(record) for record in records]))
        output_file.write('\n\nFINAL DECISION\n')
        output_file.write('\n'.join([str(f) for f in final]))
        output_file.write('\n')

    print('######################')
    print(ofile)
    print('Recall, Precision, F1, (Sieves, weight)')
    print('\n'.join([str(er) for er in results]))
    print('BEST ORDER BY PRECISION:')
    print('\n'.join([str(er) for er in best_ordered]))
    print('FINAL SELECTED SIEVES RECORD (F1)\n')
    print('\n'.join([str(record) for record in records]))
    print('\nFINAL DECISION\n')
    print('\n'.join([str(f) for f in final]))


def reset_coref_predicted_chain(topics):
    for topic in topics.topics_list:
        for mention in topic.mentions:
            mention.predicted_coref_chain = ''


def run_benchmark_on_config(topics, cdc_settings, type):
    logger.info('Running coreference resolution')
    clusters = _run_coref(topics, cdc_settings, type)

    all_mentions = from_clusters_to_mentions(clusters)
    create_chain_cluster_id(all_mentions)
    recall, precision, b3_f1 = use_scorer(all_mentions)
    return recall, precision, b3_f1


def from_clusters_to_mentions(clusters_list):
    all_mentions = list()
    for clusters in clusters_list:
        for cluster in clusters.clusters_list:
            all_mentions.extend(cluster.mentions)

    all_mentions.sort(key=lambda mention: mention.mention_index)

    return all_mentions


def load_modules(cdc_resources):
    models = list()
    models.append(None)

    # models.append(MLPRelationExtraction(str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/mlp/joint_mlp_test_model',
    #                                     ElmoEmbeddingOffline(str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/'
    #                                                                              'ecb_all_embed_bert_all_layers.pickle')))

    return models


def create_example_settings(event_config, entity_config):
    resource_location = SievesResources()

    return SievesContainerInitialization(event_config, entity_config, load_modules(resource_location))


def create_sieves_order(rel_types_weights, conf_type):
    if conf_type == 'event':
        event_config = EventSievesConfiguration()
        event_config.sieves_order = list()
        event_config.sieves_order.extend(rel_types_weights)
        entity_config = EntitySievesConfiguration()
        entity_config.run_evaluation = False
    elif conf_type == 'entity':
        entity_config = EntitySievesConfiguration()
        entity_config.sieves_order = list()
        entity_config.sieves_order.extend(rel_types_weights)
        event_config = EventSievesConfiguration()
        event_config.run_evaluation = False
    else:
        raise TypeError('no such config type:' + conf_type)

    return event_config, entity_config


def use_scorer(all_mentions):
    event_predicted_lst = [mention.predicted_coref_chain for mention in all_mentions]
    true_labels = [mention.coref_chain for mention in all_mentions]
    true_clusters_set = set(true_labels)

    labels_mapping = {}
    for label in true_clusters_set:
        labels_mapping[label] = len(labels_mapping)

    event_gold_lst = [labels_mapping[label] for label in true_labels]
    recall, precision, b3_f1 = bcubed(event_gold_lst, event_predicted_lst)
    return recall, precision, b3_f1


def create_chain_cluster_id(mentions):
    mentions_ids = dict()
    running_id = 1
    for mention in mentions:
        if mention.coref_chain in mentions_ids:
            mention.coref_chain = mentions_ids[mention.coref_chain]
        else:
            mentions_ids[mention.coref_chain] = running_id
            mention.coref_chain = running_id
            running_id += 1


def load_mentions():
    event_topics = Topics()
    event_topics.create_from_file(event_mention_json, keep_order=True)
    entity_topics = Topics()
    entity_topics.create_from_file(entity_mention_json, keep_order=True)
    return entity_topics, event_topics


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    event_mention_json = str(LIBRARY_ROOT) + '/resources/ecb/gold_json/kian/ECB_Train_Dev_Event_gold_mentions.json'
    entity_mention_json = str(LIBRARY_ROOT) + '/resources/ecb/gold_json/kian/ECB_Train_Dev_Entity_gold_mentions.json'

    event_report_out = 'data/event_result_no_computed.txt'
    entity_report_out = 'data/entity_result_no_computed.txt'

    run_banchmark()
