import pickle

from os import path

from src import LIBRARY_ROOT
from src.dataobjs.dataset import WecDataSet, EcbDataSet, DataSet, POLARITY
from src.dataobjs.mention_data import MentionData
from src.dataobjs.topics import Topics


def generate_pairs(data_set, ratio):
    event_validation_file = str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _res_file
    positive_, negative_ = data_set.get_pairwise_feat(event_validation_file, ratio, sub_topics=False)
    # positive_, negative_ = get_feat_alternative(data_set, event_validation_file)

    validate_pairs(positive_, negative_)
    basename = path.basename(path.splitext(event_validation_file)[0])
    print("Positives=" + str(len(positive_)))
    pickle.dump(positive_, open(str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + basename + "_PosPairs_Subtopic.pickle", "w+b"))
    print("Negative=" + str(len(negative_)))
    pickle.dump(negative_, open(str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + basename + "_NegPairs_Subtopic.pickle", "w+b"))
    print("Done with " + event_validation_file)


def get_feat_alternative(data_set, event_validation_file):
    topics_ = Topics()
    topics_.create_from_file(event_validation_file, keep_order=True)
    new_topics = EcbDataSet.from_ecb_subtopic_to_topic(topics_)
    positive_pairs = data_set.create_pairs(new_topics, POLARITY.POSITIVE)
    mentions = MentionData.read_mentions_json_to_mentions_data_list(event_validation_file)

    _map = dict()
    negative_pairs = list()
    for ment1 in mentions:
        for ment2 in mentions:
            if ment1.coref_chain != ment2.coref_chain:
                DataSet.check_and_add_pair(_map, negative_pairs, ment1, ment2)

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
    _res_folder = "dataset_full"
    _res_file = "WEC_Dev_Full_Event_gold_mentions_validated.json"
    _data_set = EcbDataSet()
    generate_pairs(_data_set, -1)
    # validate_pairs()
    # output_examples()
