import json
import sys

import random
from collections import namedtuple

from transformers import RobertaTokenizer

from src import LIBRARY_ROOT
from src.dataobjs.cluster import Clusters
from src.dataobjs.dataset import DataSet, WecDataSet, EcbDataSet, Split
from src.dataobjs.mention_data import MentionData
from src.dataobjs.topics import Topics
from src.utils import string_utils
from src.utils.embed_utils import EmbedTransformersGenerics
from src.utils.string_utils import StringUtils


def count_verb_mentions(split_list):
    verb_phrases = 0
    for mention in split_list:
        if StringUtils.is_verb_phrase(mention.tokens_str):
            verb_phrases += 1

    print("Total verb phrases=" + str(verb_phrases))


def calc_longest_mention_and_context(split_list, message):
    longest_mention = 0
    longest_context = 0
    for mention in split_list:
        mention_encode = _tokenizer.encode(mention.mention_context[mention.tokens_number[0]:mention.tokens_number[-1] + 1])
        # context = EmbedTransformersGenerics.extract_mention_surrounding_context(mention, 205)
        if len(mention_encode) > longest_mention:
            longest_mention = len(mention_encode)
        # if len(mention.mention_context) > longest_context:
        #     longest_context = len(context_encode)

    print(message + '_longest_mention=' + str(longest_mention))
    # print(message + '_longest_context=' + str(longest_context))


def produce_cluster_stats(clusters, message):
    singletons_count = 0
    sum_mentions = 0
    sum_mentions_no_single = 0
    all_lemmas = list()
    all_lemmas_no_single = list()
    print(message + '_Clusters=' + str(len(clusters)))
    biggest_cluster = 0
    for clust in clusters.values():
        clust_len = len(clust)
        clust_lemmas = set([ment.mention_head_lemma for ment in clust])
        if clust_len == 1:
            singletons_count += 1
        else:
            sum_mentions_no_single += clust_len
            all_lemmas_no_single.extend(clust_lemmas)

        if len(clust) > biggest_cluster:
            biggest_cluster = clust_len

        sum_mentions += len(clust)
        all_lemmas.extend(clust_lemmas)

    print(message + '_Singletons=' + str(singletons_count))
    print(message + '_Non_singleton_Clusters=' + str(len(clusters) - singletons_count))
    print(message + '_Biggest cluster=' + str(biggest_cluster))
    print(message + '_Average Ment in Clust (include singletons)=' + str(sum_mentions / len(clusters)))
    print(message + '_Average Ment in Clust (exclude singletons)=' + str(sum_mentions_no_single / (len(clusters) - singletons_count)))
    print(message + '_Average Lemmas in Clust (include singletons)=' + str(len(all_lemmas) / len(clusters)))
    print(message + '_Average Lemmas in Clust (exclude singletons)=' + str(len(all_lemmas_no_single) / (len(clusters) - singletons_count)))


def calc_single_head_lemma_cluster(ment_list, message, clus_size_thresh):
    clusters = Clusters.from_mentions_to_gold_clusters(ment_list)
    produce_cluster_stats(clusters, message)
    lemma_clust = dict()
    for clust_id, cluster in clusters.items():
        if len(cluster) > clus_size_thresh:
            if clust_id not in lemma_clust:
                lemma_clust[clust_id] = dict()
            for ment in cluster:
                ment_key = ment.mention_head_lemma.lower()
                if not ment_key in lemma_clust[clust_id]:
                    lemma_clust[clust_id][ment_key] = 0
                lemma_clust[clust_id][ment_key] += 1

    diverse_clusts = list()
    single_head_lemma_clust = 0
    for key, head_set in lemma_clust.items():
        if len(head_set) == 1:
            single_head_lemma_clust += 1
        else:
            diverse_clusts.append(head_set)

    sample = random.sample(diverse_clusts, 10)
    output_str = ""
    for sp in sample:
        for key, value in sp.items():
            output_str += key + "(" + str(value) + "), "
        output_str += "\n"

    print(message + ": single_head_lemma_clust=" + str(single_head_lemma_clust))
    print(output_str)


def extract_tp_lemma_pairs():
    readlines = open(str(LIBRARY_ROOT) + "/reports/pairs_final/TP_WEC_WEC_final_a35a3_Dev_paris.txt",
                     "r").readlines()

    sample = random.sample(readlines, 10)
    print("".join(sample))


def calc_dist_lemmas_cross(split_list, message):
    print(message + '_Mentions=' + str(len(split_list)))
    mention_length_sum = sum([len(ment.tokens_number) for ment in split_list])
    average_length = mention_length_sum / len(split_list)
    print(message + '_Average Ment Length (tokens)=' + str(average_length))

    distinct_lemmas = dict()
    distinct_lemmas_cross = dict()
    for mention in split_list:
        if mention.mention_head_lemma.lower() not in distinct_lemmas:
            distinct_lemmas[mention.mention_head_lemma.lower()] = mention

        lem_id = mention.mention_head_lemma.lower() + '_' + ''.join(filter(lambda i: i.isdigit(), str(mention.topic_id)))
        if lem_id not in distinct_lemmas_cross:
            distinct_lemmas_cross[lem_id] = set()
        distinct_lemmas_cross[lem_id].add(mention.coref_chain)

    sum_cross_clust_lem = sum([1 for clust_set in distinct_lemmas_cross.values() if len(clust_set) > 1])
    avg_cross_clust_lem = sum([len(clust_set) for clust_set in distinct_lemmas_cross.values()]) / len(distinct_lemmas_cross)

    print(message + '_Distinct Lemmas in corpus=' + str(len(distinct_lemmas)))
    print(message + '_Distinct Lemmas across clusters=' + str(sum_cross_clust_lem))
    print(message + '_Avg num of clusters with same Lemma=' + str(avg_cross_clust_lem))
    print()

    count_verb_mentions(split_list)
    # json.dump({k: v for k, v in sorted(distinct_lemmas.items(), key=lambda item: item[1])}, sys.stdout)
    print()


def cal_head_lemma_pairs(data_file, dataset, message, alpha):
    positives, negatives = dataset.get_pairwise_feat(data_file, alpha)
    same_head_pos, same_head_neg = (0, 0)
    not_same_head_pos, not_same_head_neg = (0, 0)
    for mention1, mention2 in positives:
        if mention1.mention_head_lemma == mention2.mention_head_lemma:
            same_head_pos += 1
        else:
            not_same_head_pos += 1

    for mention1, mention2 in negatives:
        if mention1.mention_head_lemma == mention2.mention_head_lemma:
            same_head_neg += 1
        else:
            not_same_head_neg += 1

    print(message + '_same_head_pos=' + str(same_head_pos))
    print(message + '_not_same_head_pos=' + str(not_same_head_pos))
    print(message + '_same_head_neg=' + str(same_head_neg))
    print(message + '_not_same_head_neg=' + str(not_same_head_neg))


def calc_tp_fp_pairs_lemma():
    f_tp = open(str(LIBRARY_ROOT) + "/reports/pairs_final/TP_WEC_WEC_final_a35a3_Dev_paris_head.txt", "r")
    tp_diff_string, tp_same_string, tp_total = generate_pair_score(f_tp)

    print("TP_TOTAL=" + str(tp_total))
    print("TP_SAME_STRING=" + str(tp_same_string))
    print("TP_DIFF_STRING=" + str(tp_diff_string))
    print("#######################################")

    f_fp = open(str(LIBRARY_ROOT) + "/reports/pairs_final/FP_WEC_WEC_final_a35a3_Dev_paris_head.txt", "r")
    fp_diff_string, fp_same_string, fp_total = generate_pair_score(f_fp)

    print("FP_TOTAL=" + str(fp_total))
    print("FP_SAME_STRING=" + str(fp_same_string))
    print("FP_DIFF_STRING=" + str(fp_diff_string))
    print("#######################################")

    f_fn = open(str(LIBRARY_ROOT) + "/reports/pairs_final/FN_WEC_WEC_final_a35a3_Dev_paris_head.txt", "r")
    fn_diff_string, fn_same_string, fn_total = generate_pair_score(f_fn)

    print("FN_TOTAL=" + str(fn_total))
    print("FN_SAME_STRING=" + str(fn_same_string))
    print("FN_DIFF_STRING=" + str(fn_diff_string))
    print("#######################################")

    f_tn = open(str(LIBRARY_ROOT) + "/reports/pairs_final/TN_WEC_WEC_final_a35a3_Dev_paris_head.txt", "r")
    tn_diff_string, tn_same_string, tn_total = generate_pair_score(f_tn)

    print("TN_TOTAL=" + str(tn_total))
    print("TN_SAME_STRING=" + str(tn_same_string))
    print("TN_DIFF_STRING=" + str(tn_diff_string))
    print("#######################################")


def cross_doc_clusters(mentions):
    MentionKey = namedtuple("MentKey", ["coref", "doc_id"])
    cross_doc_clust = dict()
    documents = set()
    for ment in mentions:
        if ment.doc_id not in documents:
            documents.add(ment.doc_id)

        ment_key = MentionKey(ment.coref_chain, ment.doc_id)
        if ment_key not in cross_doc_clust:
            cross_doc_clust[ment_key] = list()
        cross_doc_clust[ment_key].append(ment)

    cross_clusters = dict()
    for coref1, doc_id1 in cross_doc_clust.keys():
        if coref1 not in cross_clusters:
            cross_clusters[coref1] = 0
        cross_clusters[coref1] += 1

    print("Documents = " + str(len(documents)))
    print("Cross Doc Clusters = " + str(sum(1 for value in cross_clusters.values() if value > 1)))


def generate_pair_score(f):
    same_string = 0
    diff_string = 0
    total = 0
    for line in f.readlines():
        split = line.split("=")
        if len(split) == 2:
            total += 1
            if split[0].strip().lower() == split[1].strip().lower():
                same_string += 1
            else:
                diff_string += 1
    f.close()
    return diff_string, same_string, total


def create_split_stats(mentions_file, tokenizer, split):
    mentions_list = MentionData.read_mentions_json_to_mentions_data_list(mentions_file)
    if mentions_list:
        print('############# ' + split + ' ###################')
        calc_dist_lemmas_cross(mentions_list, split)
        # calc_longest_mention_and_context(mentions_list, split)
        # cal_head_lemma_pairs(mentions_file, dataset, split, 1)
        cross_doc_clusters(mentions_list)
        calc_single_head_lemma_cluster(mentions_list, split, 1)


if __name__ == '__main__':
    # _event_train = str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/train/Event_gold_mentions.json'
    _event_dev = str(LIBRARY_ROOT) + '/resources/GVC_All_gold_event_mentions.json'

    _tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    # create_split_stats(_event_train, _tokenizer, "Train")
    create_split_stats(_event_dev, _tokenizer, "ECB")

    # calc_tp_fp_pairs_lemma()
    # extract_tp_lemma_pairs()
