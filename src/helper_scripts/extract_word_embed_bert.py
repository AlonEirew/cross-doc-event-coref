import logging
import time

import numpy as np
import pickle

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
from src.utils.string_utils import StringUtils

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
    total_mentions = len(mentions)
    logger.info('Prepare to extract Bert vector from-' + str(total_mentions))
    for mention in mentions:
        total_mentions -= 1
        if mention.mention_context is not None and mention.mention_context:
            sent = ' '.join(mention.mention_context)
            if sent not in cache:
                s_t = time.time()
                bert_full_vec = get_bert_representation(mention.mention_context)
                if bert_full_vec is not None:
                    cache[sent] = bert_full_vec
                    e_t = time.time()
                    logger.info('added ' + mention.tokens_str + ', took-' + str(e_t - s_t))
                else:
                    logger.info(mention.tokens_str + ' Could not be added')

        logger.info(str(total_mentions) + ' mentions to go...')

    logger.info('Total words/contexts in vocabulary %d', len(cache))
    return cache


def get_bert_representation(in_tokens):
    """
    Embed the sentence with BERT
    """
    vocab = Vocabulary()

    tokens = StringUtils.get_tokens_from_list(in_tokens)
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
    try:
        if len(tokens) == 1:
            outputs = [outputs]
        else:
            outputs = [vec.squeeze() for vec in np.vsplit(outputs, len(tokens))]
    except:
        outputs = None
        logger.error('Failed to get bert vector')

    return outputs


def bert_dump():
    mention_files = [str(LIBRARY_ROOT) + '/resources/corpora/wiki/gold_json/WIKI_All_Event_gold_mentions.json']
    out_file = str(LIBRARY_ROOT) + '/resources/preprocessed_external_features/embedded/wiki_all_embed_bert_all_layers.pickle'

    mentions = []
    for _file in mention_files:
        mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_file))

    bert_ecb_embeddings = load_bert_for_vocab(mentions)

    with open(out_file, 'wb') as f:
        pickle.dump(bert_ecb_embeddings, f)

    logger.info('Saving dump to file-%s', out_file)


if __name__ == '__main__':
    bert_dump()
    print('Done!')
