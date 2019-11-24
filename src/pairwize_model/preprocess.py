import multiprocessing
import pickle
import time
from os import path

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics
from src.utils.bert_utils import BertPretrainedUtils

USE_CUDA = True


def extract_feature_dict(topics, bert_utils):
    result_train = dict()
    topic_count = len(topics.topics_list)
    for topic in topics.topics_list:
        mention_count = len(topic.mentions)
        for mention in topic.mentions:
            start = time.time()
            hidden, attend = bert_utils.get_mention_full_rep(mention)
            end = time.time()

            if attend is not None:
                result_train[mention.mention_id] = (hidden.cpu(), attend.cpu())
            else:
                result_train[mention.mention_id] = (hidden.cpu())

            print("To Go: Topics" + str(topic_count) + ", Mentions" + str(mention_count) + ", took-" + str((end - start)))
            mention_count -= 1
        topic_count -= 1

    return result_train


def worker(resource_file):
    name = multiprocessing.current_process().name
    print(name, 'Starting')
    bert_utils = BertPretrainedUtils(-1)

    topics = Topics()
    topics.create_from_file(resource_file, keep_order=True)
    train_feat = extract_feature_dict(topics, bert_utils)
    basename = path.basename(path.splitext(resource_file)[0])
    pickle.dump(train_feat, open(str(LIBRARY_ROOT) + "/resources/single_sent_clean_kenton/" +
                                 basename + ".pickle", "w+b"))

    print("Done with -" + basename)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    all_files = [str(LIBRARY_ROOT) + '/resources/single_sent_clean_kenton/WEC_Dev_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/single_sent_clean_kenton/WEC_Train_Event_gold_mentions.json',
                 str(LIBRARY_ROOT) + '/resources/single_sent_clean_kenton/WEC_Test_Event_gold_mentions.json'
                 ]

    jobs = list()
    for _resource_file in all_files:
        job = multiprocessing.Process(target=worker, args=(_resource_file,))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()

    print("DONE!")
