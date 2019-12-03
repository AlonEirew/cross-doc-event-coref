import logging
import multiprocessing

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.dataobjs.min_span_mention import set_mina_span
from src.utils.json_utils import write_mention_to_json
from src.utils.sqlite_utils import create_connection, select_from_validation

logger = logging.getLogger(__name__)


def run_split(split, out_file_full, out_file_single):
    # predictor = Predictor.from_path(
    #     "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz",
    #     cuda_device=0)

    connection = create_connection(str(LIBRARY_ROOT) + "/resources/EnWikiLinks_v9.db")

    name = multiprocessing.current_process().name
    print(name, 'Starting')
    mentions_full, mentions_single = extract_from_sql(connection, "Validation3", split)
    mentions_full = clean_long_mentions(mentions_full)
    mentions_single = clean_long_mentions(mentions_single)
    # extract_constituency_trees(mentions_full, predictor)
    print("Writing new full context file: " + out_file_full)
    write_mention_to_json(out_file_full, mentions_full)
    print("Writing new single sentence file: " + out_file_single)
    write_mention_to_json(out_file_single, mentions_single)
    print(name, 'Exiting')


def extract_constituency_trees(mentions, predictor):
    logging.info('Mention level constituency trees')
    mention_str = [{"sentence":  mention.tokens_str} for mention in mentions]
    keys = [mention.mention_id for mention in mentions]
    constituency_trees = []
    batch_size = 1024

    for i in range(0, len(mention_str), batch_size):
        constituency_trees.extend(predictor.predict_batch_json(mention_str[i:i+batch_size]))

    trees = {key: tree['hierplane_tree']['root'] for key, tree in zip(keys, constituency_trees)}
    logger.info('{} mentions were processed'.format(len(trees) + 1))

    set_mina_span(trees, mentions)


def get_root_of_mention(root, mention):
    for child in root['children']:
        if mention == child['word']:
            return child
        elif mention in child['word']:
            root = child

    print('No root matched for the mention: {}'.format(mention))
    return None


def clean_long_mentions(mentions_to_clean):
    new_mentions = list()
    for mention in mentions_to_clean:
        if len(mention.tokens_number) <= 7 and len(mention.mention_context) <= 75:
            new_mentions.append(mention)

    print("Total mentions with exceeding span or context cleaned-" + str(len(mentions_to_clean) - len(new_mentions)))
    return new_mentions


def extract_from_sql(connection, table_name, split, limit=-1):
    if connection is not None:
        clusters = select_from_validation(connection, table_name, split, limit=limit)
        mentions_full_context, mentions_single_sent = gen_mentions(clusters)
        mentions_full_context.sort(key=lambda mention: mention.coref_chain)
        mentions_single_sent.sort(key=lambda mention: mention.coref_chain)
        print(split + ' total mentions full context=' + str(len(mentions_full_context)))
        print(split + ' total mentions sentence context=' + str(len(mentions_single_sent)))
        return mentions_full_context, mentions_single_sent

    return None


def gen_mentions(set_clusters):
    ret_mentions_full_context = list()
    ret_mentions_single_sent = list()
    for cluster_id, cluster_ments in set_clusters.items():
        is_singleton = False
        if len(set_clusters[cluster_id]) == 1:
            is_singleton = True
        for mention in cluster_ments:
            # gen_mention_full = MentionData.read_sqlite_mention_data_line_v8(mention, gen_lemma=True, extract_valid_sent=False)
            gen_mention_full, gen_mention_single = MentionData.read_sqlite_mention_data_line_v9(mention, gen_lemma=True, extract_valid_sent=True)
            gen_mention_full.is_singleton = is_singleton
            gen_mention_single.is_singleton = is_singleton
            ret_mentions_full_context.append(gen_mention_full)
            ret_mentions_single_sent.append(gen_mention_single)

    return ret_mentions_full_context, ret_mentions_single_sent


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Output json files to write to
    validation_out_full = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Full_Event_gold_mentions.json'
    train_out_full = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Train_Full_Event_gold_mentions.json'
    test_out_full = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Full_Event_gold_mentions.json'

    validation_out_single = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Single_Event_gold_mentions.json'
    train_out_full_single = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Train_Single_Event_gold_mentions.json'
    test_out_full_single = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Single_Event_gold_mentions.json'

    # Extract the mentions from the sqlite table
    val_prc = multiprocessing.Process(target=run_split, args=('VALIDATION', validation_out_full,validation_out_single,))
    train_prc = multiprocessing.Process(target=run_split, args=('TRAIN', train_out_full, train_out_full_single,))
    test_prc = multiprocessing.Process(target=run_split, args=('TEST', test_out_full, test_out_full_single,))

    val_prc.start()
    train_prc.start()
    test_prc.start()

    val_prc.join()
    train_prc.join()
    test_prc.join()

    print('DONE!')
