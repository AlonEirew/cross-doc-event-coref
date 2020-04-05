import random
from heapq import heappush, heappop

import spacy

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics


class VisualCluster(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.mentions = set()

    def add_mention(self, start, end):
        self.mentions.add((start, end))


def from_topics_to_clusters(all_topics):
    clusters = dict()
    cluster_id_to_num = dict()
    running_num = 0
    for top in all_topics.topics_dict.values():
        for mention in top.mentions:
            if mention.coref_chain not in clusters:
                clusters[mention.coref_chain] = list()
            clusters[mention.coref_chain].append(mention)

            # mention_cluster = mention.topic_id + mention.coref_chain
            # if mention_cluster in cluster_id_to_num:
            #     mention_int_cluster_id = cluster_id_to_num[mention_cluster]
            #     clusters[mention_int_cluster_id].append(mention)
            # else:
            #     cluster_id_to_num[mention_cluster] = running_num
            #     clusters[running_num] = [mention]
            #     running_num += 1

    return clusters


def print_num_of_mentions_in_cluster(clusters):
    bucket = dict()
    for cluster in clusters.values():
        sum_men_in_clust = len(cluster)
        if sum_men_in_clust in bucket:
            bucket[sum_men_in_clust] += 1
        else:
            bucket[sum_men_in_clust] = 1

    print_dict = dict(sorted(bucket.items()))

    for key, value in print_dict.items():
        print(str(key) + ': ' + str(value))


def main(event_file):
    print('Done Loading all pairs, loading topics')
    event_topics = Topics()
    event_topics.create_from_file(event_file, keep_order=True)
    print('Done Loading topics, create stats')
    # entity_clusters = from_topics_to_clusters(entity_topics)
    event_clusters = from_topics_to_clusters(event_topics)

    print('Entity cluster distribution')
    # print_num_of_mentions_in_cluster(entity_clusters)

    print()
    print('Event cluster distribution')
    print_num_of_mentions_in_cluster(event_clusters)

    visualize_clusters(event_clusters)


def visualize_clusters(clusters):
    dispacy_obj = list()

    clus_keys = list(clusters.keys())
    random.shuffle(clus_keys)
    for cluster_id in clus_keys:
        cluster_ments = clusters.get(cluster_id)
        context_mentions = dict()
        unique_mentions_head = set()
        cluster_ments_count = 0
        for mention in cluster_ments:
            if not hasattr(mention, "manual_score"):
            # if mention.manual_score in threash:
                cluster_ments_count += 1
                unique_mentions_head.add(mention.tokens_str.lower())
                context, start, end = get_context_start_end(mention)
                if context not in context_mentions:
                    context_mentions[context] = list()

                heappush(context_mentions[context], (start, end, mention.mention_id, str(cluster_id)))

        cluster_context = ""
        ents = list()

        for context, mentions_heap in context_mentions.items():
            for i in range(len(mentions_heap)):
                ment_pair = heappop(mentions_heap)
                real_start = len(cluster_context) + 1 + ment_pair[0]
                real_end = len(cluster_context) + 1 + ment_pair[1]
                ent_label = ment_pair[2] + '(' + ment_pair[3] + ')'
                ents.append({'start': real_start, 'end': real_end, 'label': ent_label})

            cluster_context = cluster_context + '\n' + context

        if cluster_ments_count > 0:
            clust_title = str(cluster_id) + '(mentions:' + str(len(cluster_ments)) \
                          + ', unique:' + str(len(unique_mentions_head)) + ')'

            dispacy_obj.append({
                'text': cluster_context,
                'ents': ents,
                'title': clust_title
            })

    spacy.displacy.serve(dispacy_obj, style='ent', manual=True)


def get_context_start_end(mention):
    start = -1
    end = -1
    context = ""
    for i in range(len(mention.mention_context)):
        if i == mention.tokens_number[0]:
            start = len(context)

        if i == 0:
            context = mention.mention_context[i]
        else:
            context = context + ' ' + mention.mention_context[i]

        if i == int(mention.tokens_number[-1]):
            end = len(context)

    return context, start, end


if __name__ == '__main__':
    _event_file = str(LIBRARY_ROOT) + '/resources/dataset/wec/test/WEC_Test_Event_gold_mentions_validated.json'
    threash = [-1]
    main(_event_file)
