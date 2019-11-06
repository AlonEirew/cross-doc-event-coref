from transformers import BertTokenizer

from src import LIBRARY_ROOT
from src.obj.mention_data import MentionData


def calc_longest_mention(split_list, message):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    longest = 0
    for mention in split_list:
        tokens = tokenizer.encode(mention.tokens_str)
        if len(tokens) > longest:
            longest = len(tokens)

    print(message + '_longest=' + str(longest))


def calc_singletons(split_list, message):
    result_dict = dict()
    singletons_count = 0
    for mention in split_list:
        # if mention.is_singleton:
        #     singletons_count += 1
        # if mention.coref_chain.lower().startswith('singleton'):
        #     singletons_count += 1
        if mention.coref_chain in result_dict:
            result_dict[mention.coref_chain] += 1
        else:
            result_dict[mention.coref_chain] = 1

    for key, value in result_dict.items():
        if value == 1:
            singletons_count += 1

    print(message + '_Singletons=' + str(singletons_count))


if __name__ == '__main__':
    _event_train = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Train_Event_gold_mentions.json'
    _event_dev = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Dev_Event_gold_mentions.json'
    _event_test = str(LIBRARY_ROOT) + '/resources/corpora/ecb/gold_json/ECB_Test_Event_gold_mentions.json'

    _train_list = MentionData.read_mentions_json_to_mentions_data_list(_event_train)
    _dev_list = MentionData.read_mentions_json_to_mentions_data_list(_event_dev)
    _test_list = MentionData.read_mentions_json_to_mentions_data_list(_event_test)

    # calc_singletons(_train_list, "Train")
    # calc_singletons(_dev_list, "Dev")
    # calc_singletons(_test_list, "Test")

    calc_longest_mention(_train_list, "Train")
    calc_longest_mention(_dev_list, "Dev")
    calc_longest_mention(_test_list, "Test")

