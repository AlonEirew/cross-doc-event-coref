import unittest

from src.dataobjs.mention_data import MentionData
from src.utils.string_utils import StringUtils


class TestMentions(unittest.TestCase):
    def test_mention_span_align(self):
        mentions = list()
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list('test_res/Event_gold_mentions.json'))
        for mention in mentions:
            for i, tok_id in enumerate(mention.tokens_number):
                mention_text = list(zip(*StringUtils.get_tokenized_string(mention.tokens_str)))[0]
                if mention_text[i] != mention.mention_context[tok_id]:
                    raise Exception("Issue with mention-" + str(mention.mention_id))

        print("Test test_mention_span Passed!")


if __name__ == '__main__':
    unittest.main()
