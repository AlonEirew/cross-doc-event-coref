import spacy

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.sqlite_utils import create_connection


def extract_clusters(rows):
    clusters = dict()
    for mention in rows:
        cluster_id = mention[0]
        if cluster_id not in clusters:
            clusters[cluster_id] = list()

        clusters[cluster_id].append(mention)
    return clusters


def select_from_validation(conn, table_name):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param split: the validation table split
    :param coref_types: list of types to extract (default is None)
    :param limit: limit the number for rows to extract
    :return:
    """
    fields = 'coreChainId, mentionText, tokenStart, tokenEnd, extractedFromPage, ' \
             'context, PartOfSpeech, corefValue, mentionsCount, corefType, mentionId, split'

    query = "SELECT " + fields + " from " + table_name + " INNER JOIN CorefChains ON " \
                                 + table_name + ".coreChainId=CorefChains.corefId"

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    return extract_clusters(rows)


def run_process():
    event_file1 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions.json'
    event_file2 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Event_gold_mentions.json'
    event_file3 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Train_Event_gold_mentions.json'

    mentions = MentionData.read_mentions_json_to_mentions_data_list(event_file1)
    mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(event_file2))
    mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(event_file3))
    connection = create_connection(str(LIBRARY_ROOT) + "/resources/EnWikiLinks_v9.db")
    clusters = None
    if connection is not None:
        clusters = select_from_validation(connection, 'VALIDATION3')

    ment_id_cat = list()
    for mention in mentions:
        for clus_ment in clusters[mention.coref_chain]:
            if mention.mention_id == clus_ment[10]:
                ment_id_cat.append((mention.mention_id, clus_ment[9]))



if __name__ == '__main__':
    run_process()
