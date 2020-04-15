import unittest

import torch

from src import LIBRARY_ROOT
from src.dataobjs.dataset import EcbDataSet, WecDataSet
from src.dataobjs.mention_data import MentionData, MentionDataLight
from src.utils.embed_utils import EmbedTransformersGenerics, EmbeddingEnum, EmbeddingConfig


def test_mention_span():
    for mention in mentions:
        for i, tok_id in enumerate(mention.tokens_number):
            mention_text = mention.tokens_str.split(" ")
            if mention_text[i] != mention.mention_context[tok_id]:
                raise Exception("Issue with mention-" + str(mention.mention_id))

    print("Test test_mention_span Passed!")


def test_extract_mention_surrounding_context():
    context_before = ["context"]
    context_after = ["context"]
    mention_str = ["this", "is", "a", "test"]

    context_all = (context_before * 5) + mention_str + (context_after * 5)
    sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all, "None", "None", "None")
    ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics.\
        extract_mention_surrounding_context(sanity_ment, 5)

    assert((context_before * 5) == ret_context_before)
    assert((context_after * 5) == ret_context_after)
    assert(mention_str == ret_mention)

    context_all = (context_before * 5) + mention_str + (context_after * 5)
    sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all, "None",
                              "None", "None")
    ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
        extract_mention_surrounding_context(sanity_ment, 6)

    assert ((context_before * 5) == ret_context_before)
    assert ((context_after * 5) == ret_context_after)
    assert (mention_str == ret_mention)

    context_all = (context_before * 5) + mention_str + (context_after * 10)
    sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all, "None",
                              "None", "None")
    ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
        extract_mention_surrounding_context(sanity_ment, 6)

    assert ((context_before * 5) == ret_context_before)
    assert ((context_after * 6) == ret_context_after)
    assert (mention_str == ret_mention)

    context_all = (context_before * 10) + mention_str + (context_after * 5)
    sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(10, 14)), " ".join(mention_str), context_all, "None",
                              "None", "None")
    ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
        extract_mention_surrounding_context(sanity_ment, 6)

    assert ((context_before * 6) == ret_context_before)
    assert ((context_after * 5) == ret_context_after)
    assert (mention_str == ret_mention)

    context_all = mention_str + (context_after * 5)
    sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(0, 4)), " ".join(mention_str), context_all, "None",
                              "None", "None")
    ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
        extract_mention_surrounding_context(sanity_ment, 5)

    assert ([] == ret_context_before)
    assert ((context_after * 5) == ret_context_after)
    assert (mention_str == ret_mention)

    context_all = (context_before * 5) + mention_str
    sanity_ment = MentionData("-1", "-1", "-1", -1, list(range(5, 9)), " ".join(mention_str), context_all, "None",
                              "None", "None")
    ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics. \
        extract_mention_surrounding_context(sanity_ment, 5)

    assert ((context_before * 5) == ret_context_before)
    assert ([] == ret_context_after)
    assert (mention_str == ret_mention)

    for mention in mentions:
        ret_context_before, ret_mention, ret_context_after = EmbedTransformersGenerics.\
            extract_mention_surrounding_context(mention, 250)

        joined_ment_string = " ".join(ret_mention)
        if joined_ment_string != mention.tokens_str:
            print("MentionId=" + str(mention.mention_id) + ", \"" +
                  joined_ment_string + "\" != \"" + mention.tokens_str + "\"")

    print("Test test_extract_mention_surrounding_context Passed!")


def test_mention_feat_to_vec():
    for embed_type in EmbeddingEnum:
        config = EmbeddingConfig(embed_type)
        print("###### Config= " + config.embed_type.name)
        for mention in mentions:
            encoded = list(torch.tensor(config.tokenizer.encode(mention.tokens_str)[1:-1]))
            ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = \
                EmbedTransformersGenerics.mention_feat_to_vec(mention, config.tokenizer, 250, False)

            if ment1_ids.shape[1] >= 512:
                print(str(mention.mention_id) + " Has more then 512 tokens")

            from_method = list(ment1_ids[0][ment1_inx_start:ment1_inx_end])
            if encoded != from_method:
                if encoded[0] != from_method[0] or encoded[-1] != from_method[-1]:
                    print("** MentionId=" + str(mention.mention_id) + ", " +
                          str(encoded) + " != " + str(from_method))
                else:
                    print("MentionId=" + str(mention.mention_id) + ", " +
                          str(encoded) + " != " + str(from_method))

    print("Test test_mention_feat_to_vec Passed")


def test_embedding():
    for embed_type in EmbeddingEnum:
        config = EmbeddingConfig(embed_type).get_embed_utils()
        print("###### Config= " + embed_type.name)
        for mention in mentions:
            hidden1, first1_tok, last1_tok, ment1_size = config.get_mention_full_rep(mention)
            encode = config.tokenizer.encode(mention.tokens_str)[1:-1]
            if len(encode) != ment1_size:
                print("Mention=" + str(mention.mention_id) + ", " + str(len(encode)) + "!=" + str(ment1_size))


def test_compare_embeddings():
    config_roberta = EmbeddingConfig(EmbeddingEnum.ROBERTA_LARGE)
    config_roberta_seq = EmbeddingConfig(EmbeddingEnum.ROBERTA_FOR_SEQ_CLASSIFICATION)
    roberta_large = config_roberta.get_embed_utils()
    roberta_for_seq = config_roberta_seq.get_embed_utils()

    mentions_loc = MentionData.read_mentions_json_to_mentions_data_list(
        str(LIBRARY_ROOT) + '/resources/dataset_full/ecb/dev/Event_gold_mentions.json')

    for mention in mentions_loc:
        hidden1, first1_tok, last1_tok, ment1_size = roberta_large.get_mention_full_rep(mention)
        hidden2, first2_tok, last2_tok, ment2_size = roberta_for_seq.get_mention_full_rep(mention)
        if ment1_size != ment2_size:
            print("ment1_size != ment2_size")
        if hidden1[0].tolist() == hidden2[0].tolist():
            print("hidden1 == hidden2")


def test_pairs_file():
    pairs_dict = dict()
    for pair in pairs:
        if pair[0].mention_id == pair[1].mention_id:
            raise Exception("Invalid pair with same mention found")
        key1 = pair[0].mention_id + pair[1].mention_id
        key2 = pair[1].mention_id + pair[0].mention_id
        if key1 in pairs_dict or key2 in pairs_dict:
            raise Exception("pair found twice")
        else:
            pairs_dict[key1] = True
            pairs_dict[key2] = True

    print("Test Passed!")


if __name__ == '__main__':
    mentions = list()
    mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(
        str(LIBRARY_ROOT) + '/resources/dataset_full/wec/train/Event_gold_mentions_validated2.json'))
    mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(
        str(LIBRARY_ROOT) + '/resources/dataset_full/wec/test/Event_gold_mentions_validated2.json'))
    mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(
        str(LIBRARY_ROOT) + '/resources/dataset_full/wec/dev/Event_gold_mentions_validated2.json'))

    # test_mention_span()
    # test_extract_mention_surrounding_context()
    # test_mention_feat_to_vec()
    # test_embedding()
    # test_compare_embeddings()

    dataset = 'dataset_full'
    data_set = WecDataSet()
    pairs = data_set.load_pos_neg_pickle(
        str(LIBRARY_ROOT) + '/resources/' + dataset + '/' + data_set.name.lower() + '/train/Event_gold_mentions_validated2_PosPairs.pickle',
        str(LIBRARY_ROOT) + '/resources/' + dataset + '/' + data_set.name.lower() + '/train/Event_gold_mentions_validated2_NegPairs.pickle')
    test_pairs_file()

