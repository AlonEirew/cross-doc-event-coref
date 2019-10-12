# ******************************************************************************
# Copyright 2017-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************
import logging
import time

import numpy as np
import pickle

import spacy
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers.wordpiece_indexer import PretrainedBertIndexer
from allennlp.data.tokenizers import WordTokenizer
from allennlp.data.tokenizers.word_splitter import BertBasicWordSplitter
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder

from src import LIBRARY_ROOT
from src.obj.mention_data import MentionData

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


word_tokenizer = WordTokenizer()
bert_tokenizer = WordTokenizer(word_splitter=BertBasicWordSplitter())
token_indexer = PretrainedBertIndexer("bert-large-cased", do_lowercase=False)
token_embedder = PretrainedBertEmbedder("bert-large-cased", top_layer_only=False)


def load_bert_for_vocab(mentions):
    """
    Create the embedding using the cache logic in the embedding class
    Args:
        mentions:

    Returns:

    """
    cache = dict()
    for mention in mentions:
        if mention.mention_context is not None and mention.mention_context:
            sent = ' '.join(mention.mention_context)
            if sent not in cache:
                s_t = time.time()
                bert_full_vec = get_bert_representation(mention.mention_context)
                cache[sent] = bert_full_vec
                e_t = time.time()
                print('added ' + mention.tokens_str + ', took-' + str(e_t - s_t))

    logger.info('Total words/contexts in vocabulary %d', len(cache))
    return cache


def get_bert_representation(in_tokens):
    """
    Embed the sentence with BERT
    """
    vocab = Vocabulary()

    tokens = nlp.tokenizer.tokens_from_list(in_tokens)
    instance = Instance({"tokens": TextField(tokens, {"bert": token_indexer})})
    batch = Batch([instance])
    batch.index_instances(vocab)

    padding_lengths = batch.get_padding_lengths()
    tensor_dict = batch.as_tensor_dict(padding_lengths)
    bert_tokens = tensor_dict["tokens"]["bert"]
    offsets = tensor_dict["tokens"]["bert-offsets"]

    bert_vectors = token_embedder(bert_tokens, offsets=offsets)
    outputs = bert_vectors.squeeze().data.numpy()

    # Matrix -> list of vectors
    if len(tokens) == 1:
        outputs = [outputs]
    else:
        outputs = [vec.squeeze() for vec in np.vsplit(outputs, len(tokens))]

    return outputs


def bert_dump():
    out_file = str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/wiki_all_embed_bert_all_layers.pickle'
    mention_files = [str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_All_Event_gold_mentions.json']

    mentions = []
    for _file in mention_files:
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_file))

    bert_ecb_embeddings = load_bert_for_vocab(mentions)

    with open(out_file, 'wb') as f:
        pickle.dump(bert_ecb_embeddings, f)

    logger.info('Saving dump to file-%s', out_file)


if __name__ == '__main__':
    nlp = spacy.load('en')
    bert_dump()
    print('Done!')
