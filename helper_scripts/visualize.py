"""
Usage:
    preprocess_embed.py <MentionsFile> --present=<x> [--sample=<y>]

Options:
    -h --help       Show this screen.
    --present=<x>   topic/cluster - (topic relevant only to ECB+) Visualize the mentions grouped by topic or clusters
    --sample=<y>    Sample y clusters/topics to visualize [default: -1]
"""

import random
from heapq import heappush, heappop

import spacy
from docopt import docopt

from src.dataobjs.mention_data import MentionData
from src.dataobjs.topics import Topics


class VisualCluster(object):
    def __init__(self, sentence):
        self.sentence = sentence
        self.mentions = set()

    def add_mention(self, start, end):
        self.mentions.add((start, end))


def from_topics_to_clusters(all_topics):
    clusters = dict()
    for top in all_topics.topics_dict.values():
        for mention in top.mentions:
            if mention.coref_chain not in clusters:
                clusters[mention.coref_chain] = list()
            clusters[mention.coref_chain].append(mention)
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


def visualize_mentions():
    dispacy_obj = list()

    pages_dict = dict()
    for _mention in _mentions:
        if _mention.doc_id not in pages_dict:
            pages_dict[_mention.doc_id] = list()
        pages_dict[_mention.doc_id].append(_mention)

    for mentions in pages_dict.values():
        mentions.sort(key=lambda x: x.tokens_number[0])

    sampled = 1
    for doc_id, mentions in pages_dict.items():
        ents = list()
        context = ""
        for mention in mentions:
            context, start, end = get_context_start_end(mention)
            label = mention.mention_id
            ents.append({'start': start, 'end': end + 1, 'label': label})

        if _sample < 0 or 0 < sampled <= _sample:
            sampled += 1
            if ents:
                dispacy_obj.append({
                    'text': context,
                    'ents': ents,
                    'title': doc_id
                })
        else:
            break

    spacy.displacy.serve(dispacy_obj, style='ent', manual=True)


def visualize_clusters():
    event_topics = Topics()
    event_topics.create_from_file(_event_file, keep_order=True)
    clusters = from_topics_to_clusters(event_topics)

    dispacy_obj = list()
    sampled = 1
    clus_keys = list(clusters.keys())
    random.shuffle(clus_keys)
    for cluster_id in clus_keys:
        cluster_ments = clusters.get(cluster_id)
        context_mentions = dict()
        unique_mentions_head = set()
        cluster_ments_count = 0
        for mention in cluster_ments:
            cluster_ments_count += 1
            unique_mentions_head.add(mention.tokens_str.lower())
            context, start, end = get_context_start_end(mention)
            if context not in context_mentions:
                context_mentions[context] = list()
            heappush(context_mentions[context], (start, end, mention.mention_id))

        cluster_context = ""
        ents = list()

        for context, mentions_heap in context_mentions.items():
            for i in range(len(mentions_heap)):
                ment_pair = heappop(mentions_heap)
                real_start = len(cluster_context) + 1 + ment_pair[0]
                real_end = len(cluster_context) + 1 + ment_pair[1]
                ent_label = ment_pair[2]
                ents.append({'start': real_start, 'end': real_end, 'label': ent_label})

            cluster_context = cluster_context + '\n' + context

        if cluster_ments_count > 0 and (_sample < 0 or 0 < sampled <= _sample):
            sampled += 1
            clust_title = 'Cluster(' + str(clusters[cluster_id][0].coref_chain) + \
                          '), Mentions(' + str(len(cluster_ments)) + ')'

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
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(arguments)
    _event_file = arguments.get("<MentionsFile>")
    _present = arguments.get("--present")
    _sample = int(arguments.get("--sample"))
    _mentions = MentionData.read_mentions_json_to_mentions_data_list(_event_file)
    if _present.lower() == "topic":
        visualize_mentions()
    elif _present.lower() == "cluster":
        visualize_clusters()
    else:
        raise ValueError("No such presentation-" + _present)
