import pickle

import os
from os import path

from src import LIBRARY_ROOT
from src.dataobjs.dataset import WecDataSet, Split, TopicConfig, EcbDataSet


def generate_pairs():
    positive_, negative_ = _data_set.get_pairwise_feat(_event_validation_file, to_topics=_topic_config)

    validate_pairs(positive_, negative_)

    basename = path.basename(path.splitext(_event_validation_file)[0])
    dirname = os.path.dirname(_event_validation_file)
    positive_file = dirname + "/" + basename + "_PosPairs.pickle"
    negative_file = dirname + "/" + basename + "_NegPairs.pickle"

    print("Positives=" + str(len(positive_)))
    pickle.dump(positive_, open(positive_file, "w+b"))

    print("Negative=" + str(len(negative_)))
    pickle.dump(negative_, open(negative_file, "w+b"))

    print("Done with " + _event_validation_file)


def validate_pairs(pos_pairs, neg_pairs):
    for men1, men2 in neg_pairs:
        if men1.coref_chain == men2.coref_chain:
            print("NEG BUG!!!!!!!!!!")

    for men1, men2 in pos_pairs:
        if men1.coref_chain != men2.coref_chain:
            print("POS BUG!!!!!!!!!!")

    print("Validation Passed!")


if __name__ == '__main__':
    _split = Split.Test
    _ratio = -1
    _topic_config = TopicConfig.SubTopic
    _data_set = WecDataSet(ratio=_ratio, split=_split)
    _res_file = "Event_gold_mentions_clean13_validated.json"

    _event_validation_file = str(LIBRARY_ROOT) + "/resources/" + _data_set.name.lower() + \
                             "/" + _split.name.lower() + "/" + _res_file

    print("Generating pairs for file-/" + _split.name + "/" + _res_file)
    generate_pairs()
    print("Process Done!")

