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


if __name__ == '__main__':

    all_files = [str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Train_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/final_set/Min_CleanWEC_Test_Event_gold_mentions.json'
                 ]

    _bert_utils = BertPretrainedUtils(-1)

    for resource_file in all_files:
        topics = Topics()
        topics.create_from_file(resource_file, keep_order=True)
        train_feat = extract_feature_dict(topics)
        basename = path.basename(path.splitext(resource_file)[0])
        pickle.dump(train_feat, open(str(LIBRARY_ROOT) + "/resources/final_set/" +
                                     basename + ".pickle", "w+b"))

        print("Done with -" + basename)
