import sqlite3

import spacy


def select_all(conn):
    """
    Query all rows in the tasks table
    :param conn: the Connection object
    :return:
    """
    cur = conn.cursor()
    cur.execute("SELECT * FROM Validation WHERE split=\"VALIDATION\"")

    rows = cur.fetchall()

    clusters = dict()
    for mention in rows:
        cluster_id = mention[2]
        cur2 = conn.cursor()
        cur2.execute("select corefValue from CorefChains where corefId=" + str(cluster_id))
        corefVal = cur2.fetchone()
        if corefVal is not None:
            mention = mention + (corefVal[0],)
        if cluster_id not in clusters:
            clusters[cluster_id] = list()

        clusters[cluster_id].append(mention)

    return clusters


def visualize_clusters(clusters):
    dispacy_obj = list()
    for cluster_ments in clusters.values():
        ents = list()
        cluster_context = ''
        for mention in cluster_ments:
            # mention_id = mention[0]
            cluster_id = mention[2]
            mention_text = mention[3]
            token_start = mention[4]
            token_end = mention[5]
            context = mention[7]
            cluster_title = mention[9]
            context_spl = context.split(' ')
            real_tok_start = len(cluster_context) + 1
            for i in range(token_start):
                real_tok_start += len(context_spl[i]) + 1

            real_tok_end = real_tok_start
            for i in range(token_start, token_end + 1):
                real_tok_end += len(context_spl[i]) + 1

            ents.append({'start': real_tok_start, 'end': real_tok_end, 'label': str(cluster_id)})
            cluster_context = cluster_context + '\n\n' + context

        clust_title = cluster_title + ' (' + str(cluster_id) + '); Mentions:' + str(len(cluster_ments))

        dispacy_obj.append({
            'text': cluster_context,
            'ents': ents,
            'title': clust_title
        })

    spacy.displacy.serve(dispacy_obj, style='ent', manual=True)


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


def run_process():
    connection = create_connection("/Users/aeirew/workspace/DataBase/WikiLinksPersonEventFull_v8.db")
    if connection is not None:
        clusters = select_all(connection)
        visualize_clusters(clusters)


if __name__ == '__main__':
    run_process()
