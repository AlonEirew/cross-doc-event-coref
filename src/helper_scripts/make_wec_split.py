import itertools
import random

from src import LIBRARY_ROOT
from src.dataobjs.cluster import Clusters
from src.dataobjs.mention_data import MentionData
from src.helper_scripts.stats_calculation import calc_singletons
from src.utils.io_utils import write_mention_to_json


MAX = 4
TEST_DEV_PER_OF_ALL = 5


def gen_split_uncut():
    origin_mentions = MentionData.read_mentions_json_to_mentions_data_list(all_ment_file)
    all_clusters = Clusters.from_mentions_to_gold_clusters(origin_mentions)
    all_clusters = list(all_clusters.values())
    random.shuffle(all_clusters)
    dev1_split_last = int(((len(all_clusters) * TEST_DEV_PER_OF_ALL) / 100))
    print("all clust=" + str(len(all_clusters)))
    dev2_split_last = 2 * dev1_split_last
    dev1_clust = all_clusters[0:dev1_split_last]
    dev2_clust = all_clusters[dev1_split_last:dev2_split_last]
    train_clust = all_clusters[dev2_split_last:]
    print("dev1 clust=" + str(len(dev1_clust)))
    print("dev2 clust=" + str(len(dev2_clust)))
    print("train clust=" + str(len(train_clust)))
    return dev1_clust, dev2_clust, train_clust


def remove_shared_docs(dev1_clust, dev2_clust, train_clust):
    dev1_split_ment = [ment for sublist in dev1_clust for ment in sublist]
    dev2_split_ment = [ment for sublist in dev2_clust for ment in sublist]
    train_split_ment = [ment for sublist in train_clust for ment in sublist]
    # for i, clust in enumerate(all_clusters.values()):
    #     if i <= dev1_split_last:
    #         dev1_split_ment.extend(clust)
    #     elif i <= dev2_split_last:
    #         dev2_split_ment.extend(clust)
    #     else:
    #         train_split_ment.extend(clust)
    print("########### BEFORE ################")
    calc_singletons(dev1_split_ment, "dev1")
    print("################################")
    calc_singletons(dev2_split_ment, "dev2")
    print("################################")
    calc_singletons(train_split_ment, "train")
    print("################################")
    print("################################")
    # print("Before=" + str(len(dev1_split_ment)))
    # print("Before=" + str(len(dev2_split_ment)))
    # print("Before=" + str(len(train_split_ment)))
    dev1_docs_set = set([ment.doc_id for ment in dev1_split_ment])
    dev2_docs_set = set([ment.doc_id for ment in dev2_split_ment])
    if len(dev1_split_ment) > len(dev2_split_ment):
        dev1_split_ment = [ment for ment in dev1_split_ment if ment.doc_id not in dev2_docs_set]
    else:
        dev2_split_ment = [ment for ment in dev2_split_ment if ment.doc_id not in dev1_docs_set]

    train_split_ment = [ment for ment in train_split_ment if
                        ment.doc_id not in dev1_docs_set and ment.doc_id not in dev2_docs_set]
    print("########### AFTER BEFORE TRIM ################")
    calc_singletons(dev1_split_ment, "dev1")
    print("################################")
    calc_singletons(dev2_split_ment, "dev2")
    print("################################")
    calc_singletons(train_split_ment, "train")
    print("################################")
    print("################################")
    print("################################")

    return dev1_split_ment, dev2_split_ment, train_split_ment


def write_files(dev1_split_ment, dev2_split_ment, train_split_ment):
    if len(dev1_split_ment) > len(dev2_split_ment):
        write_mention_to_json(test_split_out_file, dev1_split_ment)
        write_mention_to_json(dev_split_out_file, dev2_split_ment)
    else:
        write_mention_to_json(test_split_out_file, dev2_split_ment)
        write_mention_to_json(dev_split_out_file, dev1_split_ment)
    write_mention_to_json(train_split_out_file, train_split_ment)


def set_max_same_string_mentions(origin_mentions):
    clusters = Clusters.from_mentions_to_gold_clusters(origin_mentions)
    final_mentions = list()
    for clust in clusters.values():
        clust_dict_names = dict()
        for mention in clust:
            if mention.tokens_str.lower() not in clust_dict_names:
                clust_dict_names[mention.tokens_str.lower()] = list()
            clust_dict_names[mention.tokens_str.lower()].append(mention)

        for ment_list in clust_dict_names.values():
            if len(ment_list) >= MAX:
                final_mentions.extend(list(random.sample(ment_list, MAX)))
            else:
                final_mentions.extend(ment_list)

    return final_mentions


if __name__ == '__main__':
    all_ment_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/all/Event_gold_mentions_clean9_uncut_verb.json'

    train_split_out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_clean10_v2.json'
    dev_split_out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/dev/Event_gold_mentions_clean10_v2.json'
    test_split_out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/test/Event_gold_mentions_clean10_v2.json'

    _dev1_clust, _dev2_clust, _train_clust = gen_split_uncut()
    _dev1_split_ment, _dev2_split_ment, _train_split_ment = remove_shared_docs(_dev1_clust, _dev2_clust, _train_clust)
    _dev1_split_ment = set_max_same_string_mentions(_dev1_split_ment)
    _dev2_split_ment = set_max_same_string_mentions(_dev2_split_ment)
    _train_split_ment = set_max_same_string_mentions(_train_split_ment)

    print("########### AFTER TRIM ################")
    calc_singletons(_dev1_split_ment, "dev1")
    print("################################")
    calc_singletons(_dev2_split_ment, "dev2")
    print("################################")
    calc_singletons(_train_split_ment, "train")
    print("################################")
    write_files(_dev1_split_ment, _dev2_split_ment, _train_split_ment)

    print("Dont!")
