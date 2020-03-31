from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import write_mention_to_json

if __name__ == '__main__':
    validated_in_file = str(LIBRARY_ROOT) + '/resources/validated/WEC_CLEAN_JOIN.json'

    new_files = [
        str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Full_Event_gold_mentions.json',
        # str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Dev_Event_gold_mentions.json',
        str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Full_Event_gold_mentions.json',
        # str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Test_Event_gold_mentions.json',
        str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Train_Full_Event_gold_mentions.json',
        # str(LIBRARY_ROOT) + '/resources/final_dataset/WEC_Train_Event_gold_mentions.json'
    ]

    validated_mentions = MentionData.read_mentions_json_to_mentions_data_list(validated_in_file)
    valid_coref = dict()
    for mention in validated_mentions:
        valid_coref[mention.mention_id] = True

    for res_file in new_files:
        res_mention = MentionData.read_mentions_json_to_mentions_data_list(res_file)
        clean_mentions = [mention for mention in res_mention if mention.mention_id not in valid_coref]
        final_mentions = list()
        for mention in clean_mentions:
            if len(mention.tokens_number) <= 7 and len(mention.mention_context) <= 75:
                final_mentions.append(mention)
        print("removed=" + str((len(res_mention) - len(final_mentions))) + " from file=" + res_file)
        write_mention_to_json(res_file, final_mentions)

    print("Done!!!!")