import logging
import multiprocessing

from allennlp.predictors import Predictor

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.dataobjs.min_span_mention import set_mina_span
from src.utils.json_utils import write_mention_to_json
from src.utils.sqlite_utils import create_connection, select_from_validation

logger = logging.getLogger(__name__)


def run_split(split, out_file):
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz",
        cuda_device=0)

    connection = create_connection(str(LIBRARY_ROOT) + "/resources/EnWikiLinks_v9.db")

    name = multiprocessing.current_process().name
    print(name, 'Starting')
    mentions = extract_from_sql(connection, "Validation4", split)
    mentions = clean_long_mentions(mentions)
    extract_constituency_trees(mentions, predictor)
    print("Writing new file: " + out_file)
    write_mention_to_json(out_file, mentions)
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
        if len(mention.mention_context) <= 75 and len(mention.tokens_number) <= 7:
            new_mentions.append(mention)

    print("Total mentions with exceeding span or context cleaned-" + str(len(mentions_to_clean) - len(new_mentions)))
    return new_mentions


def extract_from_sql(connection, table_name, split, limit=-1):
    if connection is not None:
        clusters = select_from_validation(connection, table_name, split, limit=limit)
        mentions = gen_mentions(clusters)
        mentions.sort(key=lambda mention: mention.coref_chain)
        print(split + ' total mentions=' + str(len(mentions)))
        return mentions

    return None


def gen_mentions(set_clusters):
    ret_mentions = list()
    for cluster_id, cluster_ments in set_clusters.items():
        is_singleton = False
        if len(set_clusters[cluster_id]) == 1:
            is_singleton = True
        for mention in cluster_ments:
            # gen_mention = MentionData.read_sqlite_mention_data_line_v8(mention, gen_lemma=True, extract_valid_sent=False)
            gen_mention = MentionData.read_sqlite_mention_data_line_v9(mention, gen_lemma=True, extract_valid_sent=True)
            gen_mention.is_singleton = is_singleton
            ret_mentions.append(gen_mention)

    return ret_mentions


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Output json files to write to
    validation_out = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC4_Dev_Event_gold_mentions.json'
    train_out = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC4_Train_Event_gold_mentions.json'
    test_out = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC4_Test_Event_gold_mentions.json'

    # Extract the mentions from the sqlite table
    val_prc = multiprocessing.Process(target=run_split, args=('VALIDATION', validation_out,))
    train_prc = multiprocessing.Process(target=run_split, args=('TRAIN', train_out,))
    test_prc = multiprocessing.Process(target=run_split, args=('TEST', test_out,))

    val_prc.start()
    train_prc.start()
    test_prc.start()

    val_prc.join()
    train_prc.join()
    test_prc.join()

    print('DONE!')
