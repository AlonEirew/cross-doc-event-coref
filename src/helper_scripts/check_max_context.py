from os import path

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics

if __name__ == '__main__':
    all_files = [str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Dev_Event_gold_mentions.json'
                 ]

    for resource_file in all_files:
        count = 0
        topics = Topics()
        topics.create_from_file(resource_file, keep_order=True)
        new_mentions = list()
        for topic in topics.topics_list:
            for mention in topic.mentions:
                if len(mention.mention_context) <= 75 and len(mention.tokens_number) <= 7:
                    pass
                else:
                    count += 1

        basename = path.basename(path.splitext(resource_file)[0])
        print("*********")
        print(basename)
        print("cleaned=" + str(count))
        print("*********")
        # write_mention_to_json(str(LIBRARY_ROOT) + "/resources/corpora/single_sent_full_context_mean/Clean" +
        #                              basename + ".json", new_mentions)
