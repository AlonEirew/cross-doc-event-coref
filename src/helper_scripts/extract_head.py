from os import path

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics
from src.utils.json_utils import write_mention_to_json

if __name__ == '__main__':
    all_files = [str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context_mean/WEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context_mean/WEC_Test_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context_mean/WEC_Train_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context_mean/ECB_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context_mean/ECB_Test_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context_mean/ECB_Train_Event_gold_mentions.json'
                 ]

    count = 0
    for resource_file in all_files:
        topics = Topics()
        topics.create_from_file(resource_file, keep_order=True)
        new_mentions = list()
        for topic in topics.topics_list:
            for mention in topic.mentions:
                if mention.mention_head is not None:
                    for num in mention.tokens_number:
                        if mention.mention_context[num] == mention.mention_head:
                            mention.tokens_str = mention.mention_head
                            mention.tokens_number = [num]
                            new_mentions.append(mention)

        basename = path.basename(path.splitext(resource_file)[0])
        write_mention_to_json(str(LIBRARY_ROOT) + "/resources/corpora/head_lemma/" +
                                     basename + ".json", new_mentions)

    print(str(count))
