import itertools
import random

from src import LIBRARY_ROOT
from src.dataobjs.cluster import Clusters
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import write_mention_to_json

if __name__ == '__main__':
    all_ment_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/all/Event_gold_mentions_clean9_uncut_verb.json'

    train_split_out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_clean10_uncut.json'
    dev_split_out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/dev/Event_gold_mentions_clean10_uncut.json'
    test_split_out_file = str(LIBRARY_ROOT) + '/resources/dataset_full/wec/test/Event_gold_mentions_clean10_uncut.json'

    origin_mentions = MentionData.read_mentions_json_to_mentions_data_list(all_ment_file)
    random.shuffle(origin_mentions)
    all_clusters = Clusters.from_mentions_to_gold_clusters(origin_mentions)

    dev1_split_last = int(((len(all_clusters) * 5) / 100))
    dev2_split_last = 2 * dev1_split_last
    dev1_split_ment = list()
    dev2_split_ment = list()
    train_split_ment = list()
    for i, clust in enumerate(all_clusters.values()):
        if len(clust) > 100:
            train_split_ment.extend(clust)
            continue
        if i <= dev1_split_last:
            dev1_split_ment.extend(clust)
        elif i <= dev2_split_last:
            dev2_split_ment.extend(clust)
        else:
            train_split_ment.extend(clust)

    print("Before=" + str(len(dev1_split_ment)))
    print("Before=" + str(len(dev2_split_ment)))
    print("Before=" + str(len(train_split_ment)))

    dev1_docs_set = set([ment.doc_id for ment in dev1_split_ment])
    dev2_docs_set = set([ment.doc_id for ment in dev2_split_ment])

    if len(dev1_split_ment) > len(dev2_split_ment):
        dev1_split_ment = [ment for ment in dev1_split_ment if ment.doc_id not in dev2_docs_set]
    else:
        dev2_split_ment = [ment for ment in dev2_split_ment if ment.doc_id not in dev1_docs_set]

    removed = 0
    train_split_ment = [ment for ment in train_split_ment if ment.doc_id not in dev1_docs_set and ment.doc_id not in dev2_docs_set]

    print("After=" + str(len(dev1_split_ment)))
    print("After=" + str(len(dev2_split_ment)))
    print("After=" + str(len(train_split_ment)))

    if len(dev1_split_ment) > len(dev2_split_ment):
        write_mention_to_json(test_split_out_file, dev1_split_ment)
        write_mention_to_json(dev_split_out_file, dev2_split_ment)
    else:
        write_mention_to_json(test_split_out_file, dev2_split_ment)
        write_mention_to_json(dev_split_out_file, dev1_split_ment)

    write_mention_to_json(train_split_out_file, train_split_ment)

    print("Dont!")
