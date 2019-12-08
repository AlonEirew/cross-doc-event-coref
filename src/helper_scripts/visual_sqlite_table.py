import spacy

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.sqlite_utils import create_connection


def extract_clusters(rows):
    clusters = dict()
    for values in rows:
        cluster_id = values[0]
        if cluster_id not in clusters:
            clusters[cluster_id] = list()

        clusters[cluster_id].append(values)
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
    fields = 'coreChainId, mentionId, corefType'

    query = "SELECT " + fields + " from " + table_name + " INNER JOIN CorefChains ON " \
                                 + table_name + ".coreChainId=CorefChains.corefId where split='TRAIN'"

    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    return extract_clusters(rows)


def get_ment_count(dic):
    count = 0
    for clust_count in dic.values():
        count += clust_count
    return count


def run_process():
    # event_file1 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions.json'
    # event_file2 = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Event_gold_mentions.json'
    event_train = str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Train_Event_gold_mentions.json'

    train_mentions = MentionData.read_mentions_json_to_mentions_data_list(event_train)
    # mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(event_file2))
    # mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(event_file3))
    connection = create_connection(str(LIBRARY_ROOT) + "/resources/EnWikiLinks_v9.db")
    clusters = None
    if connection is not None:
        clusters = select_from_validation(connection, 'Validation3')

    civil = dict()
    disaster = dict()
    accident = dict()
    sport = dict()
    award = dict()
    general = dict()
    all_dicts = {2: civil, 3: disaster, 4: accident, 6: sport, 7:award, 8: general}
    for mention in train_mentions:
        if mention.coref_chain in clusters:
            for coref, ment_id, cat in clusters[mention.coref_chain]:
                if mention.mention_id == str(ment_id):
                    if coref not in all_dicts[cat]:
                        all_dicts[cat][coref] = 0
                    all_dicts[cat][coref] += 1

    print("Civil_Clusters=" + str(len(civil)))
    print("Civil_Mentions=" + str(get_ment_count(civil)))

    print("disaster_Clusters=" + str(len(disaster)))
    print("disaster_Mentions=" + str(get_ment_count(disaster)))

    print("accident_Clusters=" + str(len(accident)))
    print("accident_Mentions=" + str(get_ment_count(accident)))

    print("sport_Clusters=" + str(len(sport)))
    print("sport_Mentions=" + str(get_ment_count(sport)))

    print("award_Clusters=" + str(len(award)))
    print("award_Mentions=" + str(get_ment_count(award)))

    print("general_Clusters=" + str(len(general)))
    print("general_Mentions=" + str(get_ment_count(general)))


if __name__ == '__main__':
    run_process()
