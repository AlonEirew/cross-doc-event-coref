import spacy

from src.obj.mention_data import MentionData
from src.utils.sqlite_utils import select_split_from_validation, create_connection, select_all_from_mentions


def visualize_clusters(clusters, read_mention_line_method):
    dispacy_obj = list()
    for cluster_ments in clusters.values():
        ents = list()
        cluster_context = ''
        cluster_title = ''
        cluster_id = ''
        for mention in cluster_ments:
            mention_data = read_mention_line_method(mention, gen_lemma=False, extract_valid_sent=False)
            cluster_title = mention[7]
            cluster_id = mention_data.coref_chain
            context_spl = mention_data.mention_context
            real_tok_start = len(cluster_context) + 1
            for i in range(mention_data.tokens_number[0]):
                real_tok_start += len(context_spl[i]) + 1

            real_tok_end = real_tok_start
            for i in range(mention_data.tokens_number[0], mention_data.tokens_number[-1] + 1):
                if i < len(context_spl):
                    real_tok_end += len(context_spl[i]) + 1

            ents.append({'start': real_tok_start, 'end': real_tok_end, 'label': str(cluster_id)})
            cluster_context = cluster_context + '\n\n' + ' '.join(mention_data.mention_context)

        clust_title = cluster_title + ' (' + str(cluster_id) + '); Mentions:' + str(len(cluster_ments))

        dispacy_obj.append({
            'text': cluster_context,
            'ents': ents,
            'title': clust_title
        })

    spacy.displacy.serve(dispacy_obj, style='ent', manual=True)


def run_process():
    connection = create_connection("/Users/aeirew/workspace/DataBase/WikiLinksPersonEventFull_v9.db")
    read_mention_line_menthon = MentionData.read_sqlite_mention_data_line_v9
    if connection is not None:
        clusters = select_split_from_validation(connection, 'VALIDATION')
        visualize_clusters(clusters, read_mention_line_menthon)


if __name__ == '__main__':
    run_process()
