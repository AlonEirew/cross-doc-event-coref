import logging
import sqlite3
from typing import List


logger = logging.getLogger(__name__)


def select_from_validation(conn, table_name, split: str = None, coref_types: List[str] = None, limit: int = -1):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param split: the validation table split
    :param coref_types: list of types to extract (default is None)
    :param limit: limit the number for rows to extract
    :return:
    """
    fields = 'coreChainId, mentionText, tokenStart, tokenEnd, extractedFromPage, ' \
             'context, PartOfSpeech, corefValue, mentionsCount, corefType, mentionId'

    if split:
        fields += ', split'

    query = "SELECT " + fields + " from " + table_name + " INNER JOIN CorefChains ON " \
                                 + table_name + ".coreChainId=CorefChains.corefId"

    where = False
    if coref_types:
        query += add_multi(where)
        where = True
        coref_types = ','.join(coref_types)
        query += " corefType in (" + coref_types + ")"
    if split:
        query += add_multi(where)
        where = True
        query += " split=\"" + split + "\""
    if limit != -1:
        query += add_multi(where)
        query += " limit " + str(limit)

    print("sql query-" + query)
    cur = conn.cursor()
    cur.execute(query)
    rows = cur.fetchall()

    return extract_clusters(rows)


def add_multi(where):
    if not where:
        return " WHERE"
    return " AND"


def select_all_from_mentions(conn, table_name="Mentions", limit=-1):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param split: the validation table split
    :param limit: limit the number for rows to extract
    :return:
    """
    fields = 'coreChainId, mentionText, tokenStart, tokenEnd, extractedFromPage, ' \
             'context, PartOfSpeech, corefValue, mentionsCount, corefType, mentionId'

    cur = conn.cursor()
    if limit == -1:
        cur.execute(
            "SELECT " + fields + " from " + table_name + " INNER JOIN CorefChains ON " + table_name + ".coreChainId=CorefChains.corefId;")
    else:
        cur.execute(
            "SELECT " + fields + " from " + table_name + " INNER JOIN CorefChains ON " + table_name + ".coreChainId=CorefChains.corefId"
            " WHERE limit " + str(limit) + ";")

    rows = cur.fetchall()

    return extract_clusters(rows)


def extract_clusters(rows):
    clusters = dict()
    for mention in rows:
        cluster_id = mention[0]
        if cluster_id not in clusters:
            clusters[cluster_id] = list()

        clusters[cluster_id].append(mention)
    return clusters


def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except sqlite3.Error as e:
        logger.error(e)

    return None
