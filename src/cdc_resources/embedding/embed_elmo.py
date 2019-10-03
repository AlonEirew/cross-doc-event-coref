import logging
import pickle
from typing import List

import numpy as np

from src.obj.mention_data import MentionDataLight

logger = logging.getLogger(__name__)


class ElmoEmbedding(object):
    def __init__(self):
        logger.info('Loading Elmo Embedding module')
        self.embeder = ELMoEmbedderTFHUB()
        self.cache = dict()
        logger.info('Elmo Embedding module lead successfully')

    def get_head_feature_vector(self, mention: MentionDataLight):
        if mention.mention_context is not None and mention.mention_context:
            sentence = ' '.join(mention.mention_context)
            return self.apply_get_from_cache(sentence, True, mention.tokens_number)

        sentence = mention.tokens_str
        return self.apply_get_from_cache(sentence, False, [])

    def apply_get_from_cache(self, sentence: str, context: bool = False, indexs: List[int] = None):
        if context and indexs is not None:
            if sentence in self.cache:
                elmo_full_vec = self.cache[sentence]
            else:
                elmo_full_vec = self.embeder.get_vector(sentence.split())
                self.cache[sentence] = elmo_full_vec

            elmo_ret_vec = self.get_mention_vec_from_sent(elmo_full_vec, indexs)
        else:
            if sentence in self.cache:
                elmo_ret_vec = self.cache[sentence]
            else:
                elmo_ret_vec = self.get_elmo_avg(sentence.split())
                self.cache[sentence] = elmo_ret_vec

        return elmo_ret_vec

    def get_avrg_feature_vector(self, tokens_str):
        if tokens_str is not None:
            return self.apply_get_from_cache(tokens_str)
        return None

    def get_elmo_avg(self, sentence):
        sentence_embedding = self.embeder.get_vector(sentence)
        return np.mean(sentence_embedding, axis=0)

    @staticmethod
    def get_mention_vec_from_sent(sent_vec, indexs):
        if len(indexs) > 1:
            elmo_ret_vec = np.mean(sent_vec[indexs[0]: indexs[-1] + 1], axis=0)
        else:
            elmo_ret_vec = sent_vec[indexs[0]]

        return elmo_ret_vec


class ElmoEmbeddingOffline(object):
    def __init__(self, dump_file):
        logger.info('Loading Elmo Offline Embedding module')

        if dump_file is not None:
            with open(dump_file, 'rb') as out:
                self.embeder = pickle.load(out)
        else:
            logger.warning('Elmo Offline without loaded embeder!')

        logger.info('Elmo Offline Embedding module lead successfully')

    def get_head_feature_vector(self, mention: MentionDataLight):
        embed = None
        if mention.mention_context is not None and mention.mention_context:
            sentence = ' '.join(mention.mention_context)
            if sentence in self.embeder:
                elmo_full_vec = self.embeder[sentence]
                return ElmoEmbedding.get_mention_vec_from_sent(
                    elmo_full_vec, mention.tokens_number)

        sentence = mention.tokens_str
        if sentence in self.embeder:
            embed = self.embeder[sentence]

        return embed

    def get_avrg_feature_vector(self, tokens_str):
        embed = None
        if tokens_str in self.embeder:
            embed = self.embeder[tokens_str]

        return embed
