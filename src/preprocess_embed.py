import multiprocessing
import pickle
import time

import os
from os import path

from src import LIBRARY_ROOT
from src.dataobjs.topics import Topics
from src.utils.embed_utils import EmbedModel

USE_CUDA = True


def extract_feature_dict(topics: Topics, embed_model):
    result_train = dict()
    topic_count = len(topics.topics_dict)
    for topic in topics.topics_dict.values():
        mention_count = len(topic.mentions)
        for mention in topic.mentions:
            start = time.time()
            hidden, first_tok, last_tok, ment_size = embed_model.get_mention_full_rep(mention)
            end = time.time()

            result_train[mention.mention_id] = (hidden.cpu(), first_tok.cpu(), last_tok.cpu(), ment_size)
            print("To Go: Topics" + str(topic_count) + ", Mentions" + str(mention_count) + ", took-" + str((end - start)))
            mention_count -= 1
        topic_count -= 1

    return result_train


def worker(resource_file, max_surrounding_contx, finetune, use_cuda):
    embed_model = EmbedModel(max_surrounding_contx=max_surrounding_contx, finetune=finetune, use_cuda=use_cuda)
    name = multiprocessing.current_process().name
    print(name, "Starting")

    basename = path.basename(path.splitext(resource_file)[0])
    dirname = os.path.dirname(resource_file)
    save_to = dirname + "/" + basename + "_roberta_large.pickle"

    topics = Topics()
    topics.create_from_file(resource_file, keep_order=True)
    train_feat = extract_feature_dict(topics, embed_model)
    pickle.dump(train_feat, open(save_to, "w+b"))

    print("Done with -" + basename)


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    _dataset_name = "wec"
    _max_surrounding_contx = 250
    _finetune = False
    _use_cuda = True

    all_files = [str(LIBRARY_ROOT) + "/resources/" + _dataset_name + "/dev/Event_gold_mentions_clean13_validated.json",
                 str(LIBRARY_ROOT) + "/resources/" + _dataset_name + "/test/Event_gold_mentions_clean13_validated.json",
                 str(LIBRARY_ROOT) + "/resources/" + _dataset_name + "/train/Event_gold_mentions_clean13.json"
                 ]

    print("Processing files-" + str(all_files))

    jobs = list()
    for _resource_file in all_files:
        job = multiprocessing.Process(target=worker, args=(_resource_file, _max_surrounding_contx, _finetune, _use_cuda))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()

    print("DONE!")