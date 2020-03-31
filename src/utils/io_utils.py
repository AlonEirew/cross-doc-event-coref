import json

import logging
import pickle
import time
from typing import List

from src import LIBRARY_ROOT

logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """load a file into a json object"""
    try:
        with open(file_path, encoding='utf-8') as small_file:
            return json.load(small_file)
    except OSError as e:
        print(e)
        print('trying to read file in blocks')
        with open(file_path, encoding='utf-8') as big_file:
            json_string = ''
            while True:
                block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
                json_string = json_string + block
                if not block:  # Reached EOF
                    break
            return json.loads(json_string)


def write_coref_scorer_results(mentions, output_file: str):
    """
    :param mentions: List[MentionData]
    :param output_file: str
    :return:
    """
    mentions.sort(key=lambda x: x.mention_index)
    output = open(output_file, 'w')
    output.write('#begin document (ECB+/ecbplus_all); part 000\n')
    for mention in mentions:
        output.write('ECB+/ecbplus_all\t' + '(' + str(mention.predicted_coref_chain) + ')\n')
    output.write('#end document')
    output.close()


def load_mentions_from_json_file(mentions_file_path: str):
    start_data_load = time.time()
    logger.info('Loading mentions from-%s', mentions_file_path)
    mentions = load_json_file(mentions_file_path)
    end_data_load = time.time()
    took_load = end_data_load - start_data_load
    logger.info('Mentions file-%s, took:%.4f sec to load', mentions_file_path, took_load)
    return mentions


def write_mention_to_json(out_file: str, mentions: List):
    mentions.sort(key=lambda x: x.mention_index)
    with open(out_file, 'w+') as output:
        json.dump(mentions, output, default=default, indent=4, sort_keys=True, ensure_ascii=False)


def load_pickle(file_path):
    with open(file_path, mode='rb') as pickle_file:
        loaded_file = pickle.load(pickle_file)
    return loaded_file


def default(o):
    return o.__dict__
