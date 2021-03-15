import unittest

from src.dataobjs.dataset import DataSet, Split
from src.dataobjs.topics import TopicConfig
from src.preprocess_gen_pairs import validate_pairs


class TestDataUtils(unittest.TestCase):
    def test_pairs_file(self):
        dataset = DataSet.get_dataset("ecb")
        positive_, negative_ = dataset.get_pairwise_feat("test/test_res/Event_gold_mentions.json", to_topics=TopicConfig.SubTopic)
        self.validate(dataset, negative_, positive_)

        dataset = DataSet.get_dataset("wec", split=Split.Train, ratio=10)
        positive_, negative_ = dataset.get_pairwise_feat("test/test_res/Event_gold_mentions.json")
        self.assertEqual(len(positive_) * 10, len(negative_))
        self.validate(dataset, negative_, positive_)

        dataset = DataSet.get_dataset("wec", split=Split.Dev)
        positive_, negative_ = dataset.get_pairwise_feat("test/test_res/Event_gold_mentions.json")
        self.validate(dataset, negative_, positive_)

    def validate(self, dataset, negative_, positive_):
        validate_pairs(positive_, negative_)
        pairs = dataset.create_features_from_pos_neg(positive_, negative_)
        pairs_dict = dict()
        for pair in pairs:
            self.assertNotEqual(pair[0].mention_id, pair[1].mention_id, "Invalid pair with same mention found")
            key1 = pair[0].mention_id + pair[1].mention_id
            key2 = pair[1].mention_id + pair[0].mention_id
            if key1 in pairs_dict or key2 in pairs_dict:
                raise Exception("pair found twice")

            pairs_dict[key1] = True
            pairs_dict[key2] = True

        print("Test Passed!")


if __name__ == '__main__':
    unittest.main()
