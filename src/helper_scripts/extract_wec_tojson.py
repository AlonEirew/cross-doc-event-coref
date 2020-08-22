import logging

from src import LIBRARY_ROOT
from src.utils.io_utils import write_mention_to_json
from src.utils.sqlite_utils import create_connection, select_from_validation, read_sqlite_mention_data_line

logger = logging.getLogger(__name__)


def extract_mentions_from_sql(out_file_full):
    connection = create_connection("/Users/aeirew/workspace/DataBase/WikiLinksExperiment.db")

    print('Starting')
    mentions_full = extract_from_sql(connection, "Mentions")
    print("Writing new full context file: " + out_file_full)
    write_mention_to_json(out_file_full, mentions_full)
    print('Done')


def clean_long_mentions(mentions_to_clean):
    new_mentions = list()
    for mention in mentions_to_clean:
        if len(mention.tokens_number) <= 7 and len(mention.mention_context) <= 75:
            new_mentions.append(mention)

    print("Total mentions with exceeding span or context cleaned-" + str(len(mentions_to_clean) - len(new_mentions)))
    return new_mentions


def extract_from_sql(connection, table_name):
    print('Extract from sqlite and convert to mentions')
    if connection is not None:
        clusters = select_from_validation(connection, table_name)
        mentions_full_context = gen_mentions(clusters)
        mentions_full_context.sort(key=lambda mention: mention.coref_chain)
        # mentions_single_sent.sort(key=lambda mention: mention.coref_chain)
        print('total mentions full context=' + str(len(mentions_full_context)))
        return mentions_full_context

    return None


def gen_mentions(set_clusters):
    ret_mentions_full_context = list()
    for cluster_id, cluster_ments in set_clusters.items():
        is_singleton = False
        if len(set_clusters[cluster_id]) == 1:
            is_singleton = True
        for mention in cluster_ments:
            gen_mention_full = read_sqlite_mention_data_line(mention)
            gen_mention_full.is_singleton = is_singleton
            ret_mentions_full_context.append(gen_mention_full)

    return ret_mentions_full_context


if __name__ == '__main__':
    train_out_full = str(LIBRARY_ROOT) + '/resources/Event_gold_mentions.json'
    extract_mentions_from_sql(train_out_full)
    print('DONE!')
