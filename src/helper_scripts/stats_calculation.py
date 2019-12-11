import random

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.dataobjs.topics import Topics
from src.utils.dataset_utils import get_feat, SPLIT, DATASET, convert_to_clusters


def calc_longest_mention_context(split_list, message):
    longest_mention = 0
    longest_context = 0
    for mention in split_list:
        if len(mention.tokens_str) > longest_mention:
            longest_mention = len(mention.tokens_number)
        if len(mention.mention_context) > longest_context:
            longest_context = len(mention.mention_context)

    print(message + '_longest_mention=' + str(longest_mention))
    print(message + '_longest_context=' + str(longest_context))


def calc_cluster_head_lemma(ment_file, message, clus_size_thresh):
    topics = Topics()
    topics.create_from_file(ment_file, True)
    clusters = convert_to_clusters(topics)
    lemma_clust = dict()
    for clust in clusters.items():
        if len(clust[1]) > clus_size_thresh:
            if clust[0] not in lemma_clust:
                lemma_clust[clust[0]] = dict()
            for ment in clust[1]:
                ment_key = ment.mention_head_lemma.lower()
                if not ment_key in lemma_clust[clust[0]]:
                    lemma_clust[clust[0]][ment_key] = 0
                lemma_clust[clust[0]][ment_key] += 1

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


def calc_singletons(split_list, message, only_validated=False):
    result_dict = dict()
    singletons_count = 0
    mentions_length = 0.0
    final_mentions_list = list()
    for mention in split_list:
        if only_validated:
            if hasattr(mention, "manual_score") and mention.manual_score >= 4:
                final_mentions_list.append(mention)
        else:
            final_mentions_list.append(mention)

    for mention in final_mentions_list:
        if mention.coref_chain in result_dict:
            result_dict[mention.coref_chain] += 1
        else:
            result_dict[mention.coref_chain] = 1
        mentions_length += len(mention.tokens_number)

    avg_in_clust = 0.0
    for key, value in sorted(result_dict.items(), key=lambda kv: kv[1], reverse=True):
        print(str(key) + "=" + str(value))
        if value == 1:
            singletons_count += 1
        else:
            avg_in_clust += value

    average_length = mentions_length / len(final_mentions_list)
    print(message + '_Singletons=' + str(singletons_count))
    print(message + '_Mentions=' + str(len(final_mentions_list)))
    print(message + '_Non_singleton_Clusters=' + str(len(result_dict.keys()) - singletons_count))
    print(message + '_Average Length=' + str(average_length))
    print(message + '_Average Ment in Clust=' + str(avg_in_clust / (len(result_dict) - singletons_count)))


def cal_head_lemma_pairs(data_file, message, alpha, dataset):
    positives, negatives = get_feat(data_file, alpha, SPLIT.Train, dataset)
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


if __name__ == '__main__':
    _event_train = str(LIBRARY_ROOT) + '/resources/validated/WEC_Test_Full_Event_gold_mentions_reduced.json'
    # _event_dev = str(LIBRARY_ROOT) + '/resources/validated/WEC_CLEAN_JOIN.json'
    # _event_test = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC_Test_Event_gold_mentions.json'

    _train_list = MentionData.read_mentions_json_to_mentions_data_list(_event_train)
    # _dev_list = MentionData.read_mentions_json_to_mentions_data_list(_event_dev)
    # _test_list = MentionData.read_mentions_json_to_mentions_data_list(_event_test)

    # cal_head_lemma_pairs(_event_train, "DEV", 1, DATASET.WEC)

    calc_singletons(_train_list, "Train", only_validated=False)
    # calc_singletons(_dev_list, "Dev")
    # calc_singletons(_test_list, "Test")
    calc_cluster_head_lemma(_event_train, "Train", 1)

    # calc_longest_mention_context(_train_list, "Train")
    # calc_longest_mention_context(_dev_list, "Dev")
    # calc_longest_mention_context(_test_list, "Test")

    # calc_tp_fp_pairs_lemma()
    # extract_tp_lemma_pairs()
