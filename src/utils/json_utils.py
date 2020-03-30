import json
import logging
import time
from typing import List

from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import load_json_file

logger = logging.getLogger(__name__)


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


def default(o):
    return o.__dict__
