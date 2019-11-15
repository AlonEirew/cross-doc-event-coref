from os import path

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics
from src.utils.json_utils import write_mention_to_json


def extract_mention_head(mention):
    if mention.mention_head is not None:
        for num in mention.tokens_number:
            if mention.mention_context[num] == mention.mention_head:
                mention.tokens_str = mention.mention_head
                mention.tokens_number = [num]
                new_mentions.append(mention)


def extract_mention_min_span(mention):
    if mention.min_span_str is not None and len(mention.min_span_str) > 0 and len(mention.min_span_ids) > 0:
        mention.tokens_str = mention.min_span_str
        mention.tokens_number = mention.min_span_ids


if __name__ == '__main__':

    extract_head = False
    extract_min_span = True

    all_files = [str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Train_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Test_Event_gold_mentions.json'
                 ]

    count = 0
    for resource_file in all_files:
        topics = Topics()
        topics.create_from_file(resource_file, keep_order=True)
        new_mentions = list()
        for topic in topics.topics_list:
            for _mention in topic.mentions:
                if extract_head:
                    extract_mention_head(_mention)
                elif extract_min_span:
                    extract_mention_min_span(_mention)

                new_mentions.append(_mention)

        basename = path.basename(path.splitext(resource_file)[0])
        write_mention_to_json(str(LIBRARY_ROOT) + "/resources/final_set/New" +
                                     basename + ".json", new_mentions)

    print(str(count))
