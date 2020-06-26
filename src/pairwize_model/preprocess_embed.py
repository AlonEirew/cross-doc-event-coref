import multiprocessing
import os
import pickle
import time

from os import path

from src import LIBRARY_ROOT
from src.dataobjs.dataset import WecDataSet, EcbDataSet
from src.dataobjs.topics import Topics
from src.utils.embed_utils import EmbedModel, EmbeddingConfig, EmbeddingEnum

USE_CUDA = True


def extract_feature_dict(topics: Topics, embed_utils):
    result_train = dict()
    topic_count = len(topics.topics_dict)
    for topic in topics.topics_dict.values():
        mention_count = len(topic.mentions)
        for mention in topic.mentions:
            start = time.time()
            hidden, first_tok, last_tok, ment_size = embed_utils.get_mention_full_rep(mention)
            end = time.time()

            result_train[mention.mention_id] = (hidden.cpu(), first_tok.cpu(), last_tok.cpu(), ment_size)
            print("To Go: Topics" + str(topic_count) + ", Mentions" + str(mention_count) + ", took-" + str((end - start)))
            mention_count -= 1
        topic_count -= 1

    return result_train


def worker(resource_file):
    embed_config = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)
    embed_utils = embed_config.get_embed_utils(max_surrounding_contx=250, finetune=False, use_cuda=True, pad=False)
    name = multiprocessing.current_process().name
    print(name, "Starting")

    basename = path.basename(path.splitext(resource_file)[0])
    dirname = os.path.dirname(resource_file)
    save_to = dirname + "/" + basename + "_" + embed_config.embed_type.name.lower() + ".pickle"

    topics = Topics()
    topics.create_from_file(resource_file, keep_order=True)
    train_feat = extract_feature_dict(topics, embed_utils)
    pickle.dump(train_feat, open(save_to, "w+b"))

    print("Done with -" + basename)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    _res_folder = "dataset_full"
    _dataset_name = WecDataSet()

    all_files = [str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _dataset_name.name.lower() +
                 "/dev/Event_gold_mentions_clean11_validated.json",
                 # str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _dataset_name.name.lower() +
                 # "/test/Event_gold_mentions_clean10.json",
                 # str(LIBRARY_ROOT) + "/resources/" + _res_folder + "/" + _dataset_name.name.lower() +
                 # "/train/Event_gold_mentions_clean10.json"
                 ]

    print("Processing files-" + str(all_files))

    jobs = list()
    for _resource_file in all_files:
        job = multiprocessing.Process(target=worker, args=(_resource_file,))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()

    print("DONE!")
