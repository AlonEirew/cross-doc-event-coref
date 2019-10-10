import json

from src import LIBRARY_ROOT
from src.obj.mention_data import MentionData
from src.utils.sqlite_utils import create_connection, select_all_from_validation


def main():
    validation_out = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Dev_Event_gold_mentions.json'
    # train_out = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Train_Event_gold_mentions.json'
    # test_out = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_Test_Event_gold_mentions.json'

    connection = create_connection("/Users/aeirew/workspace/DataBase/WikiLinksPersonEventFull_v8_single_sent.db")
    if connection is not None:
        validation_clusters = select_all_from_validation(connection, 'VALIDATION')
        # test_clusters = select_all_from_validation(connection, 'TEST')
        # train_clusters = select_all_from_validation(connection, 'TRAIN')
        validation_mentions = gen_mentions(validation_clusters)
        validation_mentions.sort(key=lambda mention: mention.coref_chain)
        print('Validation mentions=' + str(len(validation_mentions)))

        with open(validation_out, 'w+') as output:
            json.dump(validation_mentions, output, default=default, indent=4, sort_keys=True, ensure_ascii=False)


def default(o):
    return o.__dict__


def gen_mentions(set_clusters):
    ret_mentions = list()
    for cluster_id, cluster_ments in set_clusters.items():
        is_singleton = False
        if len(set_clusters[cluster_id]) == 1:
            is_singleton = True
        for mention in cluster_ments:
            gen_mention = MentionData.read_sqlite_mention_data_line_v8(mention, gen_lemma=True, extract_valid_sent=False)
            gen_mention.is_singleton = is_singleton
            ret_mentions.append(gen_mention)

    return ret_mentions


if __name__ == '__main__':
    main()
