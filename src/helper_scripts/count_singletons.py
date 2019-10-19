from src import LIBRARY_ROOT
from src.obj.mention_data import MentionData


def calc_singletons(in_file, message):
    result_dict = dict()
    singletons_count = 0
    dev_list = MentionData.read_mentions_json_to_mentions_data_list(in_file)
    for mention in dev_list:
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
    calc_singletons(_event_train, "Train")
    calc_singletons(_event_dev, "Dev")
    calc_singletons(_event_test, "Test")

