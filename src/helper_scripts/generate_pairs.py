import pickle

from os import path

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.dataobjs.topics import Topics
from src.utils.dataset_utils import get_feat, DATASET, create_pairs, from_subtopic_to_topic, POLARITY, \
    check_and_add_pair


def generate_pairs(data_set):
    event_validation_file = str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _res_file + ".json"
    positive_, negative_ = get_feat(event_validation_file, -1, data_set)
    # positive_, negative_ = get_feat_alternative(event_validation_file)

    validate_pairs(positive_, negative_)
    basename = path.basename(path.splitext(event_validation_file)[0])
    print("Positives=" + str(len(positive_)))
    pickle.dump(positive_, open(str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + basename + "_PosPairs.pickle", "w+b"))
    print("Negative=" + str(len(negative_)))
    pickle.dump(negative_, open(str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + basename + "_NegPairs.pickle", "w+b"))
    print("Done with " + event_validation_file)


def get_feat_alternative(event_validation_file):
    topics_ = Topics()
    topics_.create_from_file(event_validation_file, keep_order=True)
    new_topics = from_subtopic_to_topic(topics_)
    positive_pairs = create_pairs(new_topics, POLARITY.POSITIVE)
    mentions = MentionData.read_mentions_json_to_mentions_data_list(event_validation_file)

    _map = dict()
    negative_pairs = list()
    for ment1 in mentions:
        for ment2 in mentions:
            if ment1.coref_chain != ment2.coref_chain:
                check_and_add_pair(_map, negative_pairs, ment1, ment2)

    return positive_pairs, negative_pairs


def validate_pairs(pos_pairs, neg_pairs):
    for men1, men2 in neg_pairs:
        if men1.coref_chain == men2.coref_chain:
            print("NEG BUG!!!!!!!!!!")

    for men1, men2 in pos_pairs:
        if men1.coref_chain != men2.coref_chain:
            print("POS BUG!!!!!!!!!!")

    print("DONE!")


def get_pairs():
    event_validation_neg = str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _res_file + "_NegPairs.pickle"
    event_validation_pos = str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _res_file + "_PosPairs.pickle"
    neg_pairs = pickle.load(open(event_validation_neg, "rb"))
    pos_pairs = pickle.load(open(event_validation_pos, "rb"))
    return neg_pairs, pos_pairs


def output_examples():
    neg_pairs, pos_pairs = get_pairs()
    neg_pairs = neg_pairs[0:100]
    pos_pairs = pos_pairs[0:100]
    print("************ NEGATIVE PAIRS ****************")
    for men1, men2 in neg_pairs:
            print(men1.tokens_str + "=" + men2.tokens_str)
    print("************ NEGATIVE PAIRS ****************")
    print()
    print("********************************************")
    print()
    print("************ POSITIVE PAIRS ****************")
    for men1, men2 in pos_pairs:
        print(men1.tokens_str + "=" + men2.tokens_str)
    print("************ POSITIVE PAIRS ****************")


if __name__ == '__main__':
    _res_folder = "final_dataset"
    _res_file = "WEC_Test_Event_gold_mentions"
    _data_set = DATASET.ECB
    generate_pairs(_data_set)
    # validate_pairs()
    # output_examples()
