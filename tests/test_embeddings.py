import unittest

import torch

from src.dataobjs.mention_data import MentionData
from src.utils.embed_utils import EmbedTransformersGenerics


class TestEmbeddings(unittest.TestCase):
    def test_extract_mention_surrounding_context(self):
        context_before = ["context"]
        context_after = ["context"]
        mention_str = ["this", "is", "a", "test"]

        context_all1 = (context_before * 5) + mention_str + (context_after * 5)
        sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all1, "None", "None", "None")
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics.\
            extract_mention_surrounding_context(sanity_ment)

        self.assertEqual((context_before * 5), ret_context_before)
        self.assertEqual((context_after * 5), ret_context_after)
        self.assertEqual(mention_str, ret_mention)

        context_all2 = (context_before * 5) + mention_str + (context_after * 5)
        sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all2, "None", "None", "None")
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
            extract_mention_surrounding_context(sanity_ment)

        self.assertEqual((context_before * 5), ret_context_before)
        self.assertEqual((context_after * 5), ret_context_after)
        self.assertEqual(mention_str, ret_mention)

        context_all3 = (context_before * 5) + mention_str + (context_after * 10)
        sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all3, "None", "None", "None")
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
            extract_mention_surrounding_context(sanity_ment)

        self.assertEqual((context_before * 5), ret_context_before)
        self.assertEqual((context_after * 6), ret_context_after)
        self.assertEqual(mention_str, ret_mention)

        context_all4 = (context_before * 10) + mention_str + (context_after * 5)
        sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(10, 14)), " ".join(mention_str), context_all4, "None", "None", "None")
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
            extract_mention_surrounding_context(sanity_ment)

        self.assertEqual((context_before * 6), ret_context_before)
        self.assertEqual((context_after * 5), ret_context_after)
        self.assertEqual(mention_str, ret_mention)

        context_all5 = mention_str + (context_after * 5)
        sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(0, 4)), " ".join(mention_str), context_all5, "None",
                                  "None", "None")
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
            extract_mention_surrounding_context(sanity_ment)

        self.assertEqual([], ret_context_before)
        self.assertEqual((context_after * 5), ret_context_after)
        self.assertEqual(mention_str, ret_mention)

        context_all6 = (context_before * 5) + mention_str
        sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all6, "None",
                                  "None", "None")
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
            extract_mention_surrounding_context(sanity_ment)

        self.assertEqual((context_before * 5), ret_context_before)
        self.assertEqual([], ret_context_after)
        self.assertEqual(mention_str, ret_mention)

        mentions = list()
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list('tests/test_res/Event_gold_mentions.json'))
        for mention in mentions:
            ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics.extract_mention_surrounding_context(mention)

            self.fixCat(["'s", ",", "'"], ret_mention)

            joined_ment_string = " ".join(ret_mention)
            if joined_ment_string != mention.tokens_str:
                print("MentionId=" + str(mention.mention_id) + ", \"" +
                      joined_ment_string + "\" != \"" + mention.tokens_str + "\"")

        print("Test test_extract_mention_surrounding_context Passed!")

    def test_mention_feat_to_vec(self):
        mentions = list()
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list('tests/test_res/Event_gold_mentions.json'))
        config = EmbedTransformersGenerics(max_surrounding_contx=250, use_cuda=False)
        for mention in mentions:
            encoded = list(torch.tensor(config.tokenizer.encode(mention.tokens_str)[1:-1]))
            decoded = config.tokenizer.decode(encoded)

            ment1_ids, ment1_inx_start, ment1_inx_end = config.mention_feat_to_vec(mention)
            self.assertTrue(ment1_ids.shape[1] < 512, str(mention.mention_id) + " Has more then 512 tokens")

            from_method = list(ment1_ids[0][ment1_inx_start:ment1_inx_end])
            decoded_from_method = config.tokenizer.decode(from_method)
            if decoded != decoded_from_method:
                if encoded[0] != from_method[0] or encoded[-1] != from_method[-1]:
                    print("** MentionId=" + str(mention.mention_id) + ", " +
                          str(decoded) + " != " + str(decoded_from_method))
                else:
                    print("MentionId=" + str(mention.mention_id) + ", " +
                          str(decoded) + " != " + str(decoded_from_method))

        print("Test test_mention_feat_to_vec Passed")

    @staticmethod
    def fixCat(strs, tok_list):
        for str_to_cat in strs:
            if str_to_cat in tok_list:
                s_ind = tok_list.index(str_to_cat)
                new_val = tok_list[s_ind - 1] + tok_list[s_ind]
                tok_list[s_ind - 1] = new_val
                tok_list.remove(str_to_cat)
        return tok_list


if __name__ == '__main__':
    unittest.main()
