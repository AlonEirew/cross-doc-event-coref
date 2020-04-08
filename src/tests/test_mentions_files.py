import unittest

from src import LIBRARY_ROOT
from src.dataobjs.mention_data import MentionData


def test_mention_span():
    mentions = MentionData.read_mentions_json_to_mentions_data_list(str(LIBRARY_ROOT) +
                   '/resources/dataset_full/wec/dev/Event_gold_mentions.json')

    for mention in mentions:
        for i, tok_id in enumerate(mention.tokens_number):
            mention_text = mention.tokens_str.split(" ")
            if mention_text[i] != mention.mention_context[tok_id]:
                raise Exception("Issue with mention-" + str(mention.mention_id))


if __name__ == '__main__':
    test_mention_span()

