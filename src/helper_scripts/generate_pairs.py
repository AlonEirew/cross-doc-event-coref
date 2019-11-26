import pickle

from os import path

from src import LIBRARY_ROOT
from src.utils.dataset_utils import get_feat, DATASET


def generate_pairs():
    event_validation_file = str(
        LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/WEC_Train_Event_gold_mentions.json"
    positive_, negative_ = get_feat(event_validation_file, 50, DATASET.WEC)
    basename = path.basename(path.splitext(event_validation_file)[0])
    print("Positives=" + str(len(positive_)))
    pickle.dump(positive_, open(str(LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/" +
                                basename + "_PosPairs.pickle", "w+b"))
    print("Negative=" + str(len(negative_)))
    pickle.dump(negative_, open(str(LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/" +
                                basename + "_NegPairs.pickle", "w+b"))
    print("Done with " + event_validation_file)


def validate_pairs():
    event_validation_neg = str(
        LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/ECB_Train_Event_gold_mentions_NegPairs.pickle"

    event_validation_pos = str(
        LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/ECB_Train_Event_gold_mentions_PosPairs.pickle"

    neg_pairs = pickle.load(open(event_validation_neg, "rb"))
    pos_pairs = pickle.load(open(event_validation_pos, "rb"))

    for men1, men2 in neg_pairs:
        if men1.coref_chain == men2.coref_chain:
            print("NEG BUG!!!!!!!!!!")

    for men1, men2 in pos_pairs:
        if men1.coref_chain != men2.coref_chain:
            print("POS BUG!!!!!!!!!!")

    print("DONE!")


if __name__ == '__main__':
    # generate_pairs()
    validate_pairs()
