import json
import logging
import re
from collections import namedtuple

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData
from src.utils.io_utils import write_mention_to_json


def main():
    auto_conll_in_file = str(LIBRARY_ROOT) + '/resources/gvc/gold.conll'
    # auto_conll_in_file = 'data/external/yang/debug_conll'
    out_all = str(LIBRARY_ROOT) + '/resources/gvc/GVC_All_gold_event_mentions.json'

    mentions = parse_to_douments(auto_conll_in_file)
    write_mention_to_json(out_all, mentions)


def parse_to_douments(file_path):
    LocMention = namedtuple("LocMention", ["doc_id", "chain", "text", "toks_id"])
    all_mentions = list()
    half_backed_mentions = list()

    token_count_id = -1
    tokens_numbers = list()
    tokens_str = list()
    doc_context = list()
    with open(file_path, 'r') as doc_file:
        for line in doc_file.readlines():
            if line.strip().startswith('#begin document'):
                continue
            if line.strip().startswith('#end document'):
                for half_backed in half_backed_mentions:
                    all_mentions.append(
                        MentionData(None, '-1', half_backed[0], -1, half_backed[3], half_backed[2],
                            doc_context, None, None, half_backed[1]))

                token_count_id = -1
                doc_context = list()
                half_backed_mentions.clear()
                tokens_numbers = list()
                tokens_str = list()
                continue
            elif line.strip() == '':
                continue

            split_line = line.split('\t')
            token_text = split_line[1].strip()
            if token_text == 'NEWLINE':
                continue

            token_count_id += 1
            doc_id = split_line[0].split('.')[0].strip()
            coref_chain = split_line[-1].strip()

            if re.match(r'\([0-9]+\)', coref_chain):
                half_backed_mentions.append(LocMention(
                    doc_id, re.sub(r'[\(\)]', '', coref_chain), token_text, [token_count_id]))
            elif re.match(r'\b[0-9]+\)', coref_chain):
                tokens_str.append(token_text)
                tokens_numbers.append(token_count_id)
                half_backed_mentions.append(LocMention(
                    doc_id, re.sub(r'[\(\)]', '', coref_chain), " ".join(tokens_str), tokens_numbers))
                tokens_str = list()
                tokens_numbers = list()
            elif re.match(r'\(?[0-9]+\b', coref_chain):
                tokens_str.append(token_text)
                tokens_numbers.append(token_count_id)

            doc_context.append(token_text)

    return all_mentions


def default(o):
    return o.__dict__


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    main()
