import pickle
import time
from os import path

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics
from src.utils.bert_utils import BertPretrainedUtils

USE_CUDA = True


def extract_feature_dict(topics):
    result_train = dict()
    topic_count = len(topics.topics_list)
    for topic in topics.topics_list:
        mention_count = len(topic.mentions)
        for mention in topic.mentions:
            start = time.time()
            hidden, attend = _bert_utils.get_mention_mean_rep(mention)
            end = time.time()
            result_train[mention.mention_id] = (hidden.cpu(), attend.cpu())
            print("To Go: Topics" + str(topic_count) + ", Mentions" + str(mention_count) + ", took-" + str((end - start)))
            mention_count -= 1
        topic_count -= 1

    return result_train


def replace_with_mini_spans(topics):
    for topic in topics.topics_list:
        for mention in topic.mentions:
            if mention.min_span_str is not None and len(mention.min_span_str) > 0:
                mention.tokens_str = mention.min_span_str
                mention.tokens_number = mention.min_span_ids


def replace_with_head_spans(topics):
    for topic in topics.topics_list:
        for mention in topic.mentions:
            if mention.mention_head is not None:
                for num in mention.tokens_number:
                    if mention.mention_context[num] == mention.mention_head:
                        mention.tokens_str = mention.mention_head
                        mention.tokens_number = [num]


if __name__ == '__main__':
    use_mini_span = False
    use_head_span = True

    all_files = [str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context/WEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context/WEC_Test_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context/WEC_Train_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context/ECB_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context/ECB_Test_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/corpora/single_sent_full_context/ECB_Train_Event_gold_mentions.json'
                 ]

    _bert_utils = BertPretrainedUtils(-1)

    for resource_file in all_files:
        topics = Topics()
        topics.create_from_file(resource_file, keep_order=True)
        if use_mini_span:
            replace_with_mini_spans(topics)

        if use_head_span:
            replace_with_head_spans(topics)

        train_feat = extract_feature_dict(topics)
        basename = path.basename(path.splitext(resource_file)[0])
        pickle.dump(train_feat, open(str(LIBRARY_ROOT) + "/resources/corpora/head_lemma/" +
                                     basename + ".pickle", "w+b"))

        print("Done with -" + basename)
