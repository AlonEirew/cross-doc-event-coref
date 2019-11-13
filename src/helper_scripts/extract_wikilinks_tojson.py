from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.json_utils import write_mention_to_json
from src.utils.sqlite_utils import create_connection, select_from_validation


def main():
    connection = create_connection("/Users/aeirew/workspace/DataBase/EnWikiLinks_v9.db")

    validation_out = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WEC_Dev_Event_gold_mentions.json'
    train_out = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WEC_Train_Event_gold_mentions.json'
    test_out = str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WEC_Test_Event_gold_mentions.json'

    extract_and_create_json(connection, validation_out, 'VALIDATION')
    extract_and_create_json(connection, train_out, 'TRAIN')
    extract_and_create_json(connection, test_out, 'TEST')


def extract_and_create_json(connection, out_file, split):
    if connection is not None:
        clusters = select_from_validation(connection, split)
        mentions = gen_mentions(clusters)
        mentions.sort(key=lambda mention: mention.coref_chain)
        print(split + ' mentions=' + str(len(mentions)))
        write_mention_to_json(out_file, mentions)


def gen_mentions(set_clusters):
    ret_mentions = list()
    for cluster_id, cluster_ments in set_clusters.items():
        is_singleton = False
        if len(set_clusters[cluster_id]) == 1:
            is_singleton = True
        for mention in cluster_ments:
            # gen_mention = MentionData.read_sqlite_mention_data_line_v8(mention, gen_lemma=True, extract_valid_sent=False)
            gen_mention = MentionData.read_sqlite_mention_data_line_v9(mention, gen_lemma=True, extract_valid_sent=False)
            gen_mention.is_singleton = is_singleton
            ret_mentions.append(gen_mention)

    return ret_mentions


if __name__ == '__main__':
    main()
