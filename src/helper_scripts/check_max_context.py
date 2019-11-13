from transformers import BertTokenizer

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    all_files = [str(LIBRARY_ROOT) + '/resources/corpora/bkp_single_sent/WEC_Train_Event_gold_mentions.json',
                 # str(LIBRARY_ROOT) + '/resources/corpora/bkp_single_sent/WEC_Dev_Event_gold_mentions.json',
                 # str(LIBRARY_ROOT) + '/resources/corpora/bkp_single_sent/WEC_Test_Event_gold_mentions.json',
                 # str(LIBRARY_ROOT) + '/resources/corpora/bkp_single_sent/ECB_Train_Event_gold_mentions.json',
                 # str(LIBRARY_ROOT) + '/resources/corpora/bkp_single_sent/ECB_Dev_Event_gold_mentions.json',
                 # str(LIBRARY_ROOT) + '/resources/corpora/bkp_single_sent/ECB_Test_Event_gold_mentions.json'
                 ]

    count = 0
    for resource_file in all_files:
        topics = Topics()
        topics.create_from_file(resource_file, keep_order=True)
        topic = topics.topics_list[0]
        # new_mentions = [mention for mention in topic.mentions if len(mention.mention_context) < 512]
        # write_mention_to_json(resource_file + '_', new_mentions)
        # filter_mentions = [mention for mention in topic.mentions if len(mention.mention_context) >= 512]
        # print(str(len(filter_mentions)))
        for mention in topic.mentions:
            context = mention.mention_context
            if len(context) >= 512:
                print("******" + str(len(context)) + "*******")
                print(resource_file)
                print("MentionId=" + mention.mention_id)
                print("Mention Coref=" + str(mention.coref_chain))
                print("Mention Context=" + " ".join(context))
                print("*************")
                count += 1

    print(str(count))
