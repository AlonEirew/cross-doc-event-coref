from heapq import heappush, heappop

import spacy

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.string_utils import StringUtils


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


def visualize_mentions(mentions_dict, pairs_list):
    dispacy_obj = list()
    for pair in pairs_list:
        ment1 = mentions_dict[pair[1]]
        ment2 = mentions_dict[pair[3]]
        ents = list()
        context1, start1, end1 = get_context_start_end(ment1)
        ents.append({'start': start1, 'end': end1 + 1, 'label': "M1"})
        context2, start2, end2 = get_context_start_end(ment2)
        ents.append({'start': start2, 'end': end2 + 1, 'label': "M2"})

        dispacy_obj.append({
            'text': context1 + "\n\n" + context2,
            'ents': ents,
            'title': ment1.mention_id + "||" + ment2.mention_id + "||" + pair[4]
        })

    spacy.displacy.serve(dispacy_obj, style='ent', manual=True)


def visualize_clusters(mentions_dict, pairs_list):
    diff_lemmas = 0
    dispacy_obj = list()
    for pair in pairs_list:
        ment1 = mentions_dict[pair[1]]
        ment2 = mentions_dict[pair[3]]
        if ment1.mention_head_lemma == ment2.mention_head_lemma:
            continue

        if not set(ment1.tokens_str.split(" ")).isdisjoint(ment2.tokens_str.split(" ")):
            continue

        diff_lemmas += 1
        mentions = [ment1, ment2]
        context_mentions = dict()
        unique_mentions_head = set()
        cluster_ments_count = 0
        for mention in mentions:
            cluster_ments_count += 1
            unique_mentions_head.add(mention.tokens_str.lower())
            context, start, end = get_context_start_end(mention)
            if context not in context_mentions:
                context_mentions[context] = list()

            heappush(context_mentions[context], (start, end, "ID:" + mention.mention_id))

        cluster_context = ""
        ents = list()

        for context, mentions_heap in context_mentions.items():
            for i in range(len(mentions_heap)):
                ment_pair = heappop(mentions_heap)
                real_start = len(cluster_context) + 1 + ment_pair[0]
                real_end = len(cluster_context) + 1 + ment_pair[1]
                ent_label = '(' + ment_pair[2] + ')' #+ '(' + ment_pair[3] + ')'
                ents.append({'start': real_start, 'end': real_end, 'label': ent_label})

            cluster_context = cluster_context + '\n' + context

        dispacy_obj.append({
            'text': cluster_context,
            'ents': ents,
            'title': ment1.mention_id + "||" + ment2.mention_id + "||" + pair[4]
        })

    print("diff Lemmas=" + str(diff_lemmas))
    spacy.displacy.serve(dispacy_obj, style='ent', manual=True)


def load_pairs(pairs_file):
    readlines = open(pairs_file, 'r').readlines()
    pairs_list = list()
    for line in readlines:
        line_split = line.split("||")
        pairs_list.append(line_split)

    return pairs_list


def load_mentions(mention_file):
    ment_dict = dict()
    mentions = MentionData.read_mentions_json_to_mentions_data_list(mention_file)
    for ment in mentions:
        ment_dict[ment.mention_id] = ment

    return ment_dict


if __name__ == '__main__':
    mentions_file = str(LIBRARY_ROOT) + "/resources/wec/dev/Event_gold_mentions.json"
    pairs_file = str(LIBRARY_ROOT) + "/reports/pairs_final/wec/TP_WEC_WEC_valid_dev_260620_roberta_large_10iter_5_Dev_paris.txt"
    print("Generating pairs for file-/" + pairs_file)
    print("Mentions file-/" + mentions_file)
    pairs_list = load_pairs(pairs_file)
    ment_dict = load_mentions(mentions_file)
    visualize_clusters(ment_dict, pairs_list)
    print("Process Done!")
