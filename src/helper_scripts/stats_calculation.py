from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.dataset_utils import get_feat, SPLIT, DATASET


def calc_longest_mention_context(split_list, message):
    longest_mention = 0
    longest_context = 0
    for mention in split_list:
        if len(mention.tokens_str) > longest_mention:
            longest_mention = len(mention.tokens_str)
        if len(mention.mention_context) > longest_context:
            longest_context = len(mention.mention_context)

    print(message + '_longest_mention=' + str(longest_mention))
    print(message + '_longest_context=' + str(longest_context))


def calc_singletons(split_list, message):
    result_dict = dict()
    singletons_count = 0
    for mention in split_list:
        if mention.coref_chain in result_dict:
            result_dict[mention.coref_chain] += 1
        else:
            result_dict[mention.coref_chain] = 1

    for key, value in result_dict.items():
        if value == 1:
            singletons_count += 1

    print(message + '_Singletons=' + str(singletons_count))
    print(message + '_Mentions=' + str(len(split_list)))
    print(message + '_Clusters=' + str(len(result_dict.keys())))


def cal_head_lemma_pairs(data_file, message, alpha, dataset):
    positives, negatives = get_feat(data_file, alpha, SPLIT.TRAIN, dataset)
    same_head_pos, same_head_neg = (0, 0)
    not_same_head_pos, not_same_head_neg = (0, 0)
    for mention1, mention2 in positives:
        if mention1.mention_head == mention2.mention_head:
            same_head_pos += 1
        else:
            not_same_head_pos += 1

    for mention1, mention2 in negatives:
        if mention1.mention_head == mention2.mention_head:
            same_head_neg += 1
        else:
            not_same_head_neg += 1

    print(message + '_same_head_pos=' + str(same_head_pos))
    print(message + '_not_same_head_pos=' + str(not_same_head_pos))
    print(message + '_same_head_neg=' + str(same_head_neg))
    print(message + '_not_same_head_neg=' + str(not_same_head_neg))


if __name__ == '__main__':
    # _event_train = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC_Train_Event_gold_mentions.json'
    _event_dev = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC_Dev_Event_gold_mentions.json'
    # _event_test = str(LIBRARY_ROOT) + '/resources/final_set_clean_min/WEC_Test_Event_gold_mentions.json'

    # _train_list = MentionData.read_mentions_json_to_mentions_data_list(_event_train)
    # _dev_list = MentionData.read_mentions_json_to_mentions_data_list(_event_dev)
    # _test_list = MentionData.read_mentions_json_to_mentions_data_list(_event_test)

    cal_head_lemma_pairs(_event_dev, "DEV", 1, DATASET.WEC)

    # calc_singletons(_train_list, "Train")
    # calc_singletons(_dev_list, "Dev")
    # calc_singletons(_test_list, "Test")

    # calc_longest_mention_context(_train_list, "Train")
    # calc_longest_mention_context(_dev_list, "Dev")
    # calc_longest_mention_context(_test_list, "Test")

