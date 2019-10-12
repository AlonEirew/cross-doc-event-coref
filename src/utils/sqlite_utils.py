import sqlite3


def select_split_from_validation(conn, split, limit=-1):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :param split: the validation table split
    :param limit: limit the number for rows to extract
    :return:
    """
    fields = 'coreChainId, mentionText, tokenStart, tokenEnd, extractedFromPage, ' \
             'context, PartOfSpeech, corefValue, mentionsCount, corefType, mentionId, split'

    cur = conn.cursor()
    if limit == -1:
        cur.execute(
            "SELECT " + fields + " from Validation INNER JOIN CorefChains ON"
            " Validation.coreChainId=CorefChains.corefId WHERE split=\"" + split + "\";")
    else:
        cur.execute(
            "SELECT " + fields + " from Validation INNER JOIN CorefChains"
            " ON Validation.coreChainId=CorefChains.corefId WHERE split=\"" + split + "\" limit " + str(limit) + ";")

    rows = cur.fetchall()

    return extract_clusters(rows)


def select_all_from_mentions(conn, limit=-1):
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
            "SELECT " + fields + " from Mentions INNER JOIN CorefChains ON Mentions.coreChainId=CorefChains.corefId;")
    else:
        cur.execute(
            "SELECT " + fields + " from Mentions INNER JOIN CorefChains ON Mentions.coreChainId=CorefChains.corefId"
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
        print(e)

    return None
