import json
import logging
from typing import List

from src.obj.mention_data import MentionData
from src.utils.string_utils import StringUtils

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def load_json_file(file_path):
    """load a file into a json object"""
    try:
        with open(file_path) as small_file:
            return json.load(small_file)
    except OSError as e:
        print(e)
        print('trying to read file in blocks')
        with open(file_path) as big_file:
            json_string = ''
            while True:
                block = big_file.read(64 * (1 << 20))  # Read 64 MB at a time;
                json_string = json_string + block
                if not block:  # Reached EOF
                    break
            return json.loads(json_string)


def load_mentions_vocab_from_files(mentions_files, filter_stop_words=False):
    logger.info('Loading mentions files...')
    mentions = []
    for _file in mentions_files:
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_file))

    return load_mentions_vocab(mentions, filter_stop_words)


def load_mentions_vocab(mentions, filter_stop_words=False):
    vocab = extract_vocab(mentions, filter_stop_words)
    logger.info('Done loading mentions files...')
    return vocab


def extract_vocab(mentions: List[MentionData], filter_stop_words: bool) -> List[str]:
    """
    Extract Head, Lemma and mention string from all mentions to create a list of string vocabulary
    Args:
        mentions:
        filter_stop_words:

    Returns:

    """
    vocab = set()
    for mention in mentions:
        head = mention.mention_head
        head_lemma = mention.mention_head_lemma
        tokens_str = mention.tokens_str
        if not filter_stop_words:
            vocab.add(head)
            vocab.add(head_lemma)
            vocab.add(tokens_str)
        else:
            if not StringUtils.is_stop(head):
                vocab.add(head)
            if not StringUtils.is_stop(head_lemma):
                vocab.add(head_lemma)
            if not StringUtils.is_stop(tokens_str):
                vocab.add(tokens_str)
    vocab_set = list(vocab)
    return vocab_set
