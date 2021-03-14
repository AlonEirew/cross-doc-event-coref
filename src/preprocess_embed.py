"""
Usage:
    preprocess_embed.py <File> [<File2>] [<File3>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--max=<x>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--cuda=<y>]
    preprocess_embed.py <File> [<File2>] [<File3>] [--max=<x>] [--cuda=<y>]

Options:
    -h --help     Show this screen.
    --max=<x>   Maximum surrounding context [default: 250]
    --cuda=<y>  True/False - Whether to use cuda device or not [default: True].

"""

import multiprocessing
import pickle
import time

import os
from os import path

from docopt import docopt

from dataobjs.topics import Topics
from utils.embed_utils import EmbedModel


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


def worker(resource_file, max_surrounding_contx, use_cuda):
    embed_model = EmbedModel(max_surrounding_contx=max_surrounding_contx, use_cuda=use_cuda)
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
    arguments = docopt(__doc__, argv=None, help=True, version=None, options_first=False)
    print(arguments)
    _file1 = arguments.get("<File>")
    _file2 = arguments.get("<File2>")
    _file3 = arguments.get("<File3>")
    _max_surrounding_contx = int(arguments.get("--max"))
    _use_cuda = True if arguments.get("--cuda").lower() == "true" else False

    _all_files = list()
    if _file1:
        _all_files.append(_file1)
    if _file2:
        _all_files.append(_file2)
    if _file3:
        _all_files.append(_file3)

    print("Processing files-" + str(_all_files))

    jobs = list()
    for _resource_file in _all_files:
        job = multiprocessing.Process(target=worker, args=(_resource_file, _max_surrounding_contx, _use_cuda))
        jobs.append(job)
        job.start()

    for job in jobs:
        job.join()

    print("DONE!")
