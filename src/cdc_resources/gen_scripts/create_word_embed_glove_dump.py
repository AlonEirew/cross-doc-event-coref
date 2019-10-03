import argparse
import logging
import pickle

import numpy as np

from nlp_architect.models.cross_doc_coref.system.cdc_utils import load_mentions_vocab_from_files
from nlp_architect.utils import io

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='Create GloVe dataset only dump')

parser.add_argument('--glove', type=str, help='glove db file', required=True)

parser.add_argument('--mentions', type=str, help='mentions file', required=True)

parser.add_argument('--output', type=str, help='location were to create dump file', required=True)

args = parser.parse_args()


def load_glove_for_vocab(glove_filename, vocabulary):
    vocab = []
    embd = []
    with open(glove_filename) as glove_file:
        for line in glove_file:
            row = line.strip().split(' ')
            word = row[0]
            if word in vocabulary:
                vocab.append(word)
                embd.append(row[1:])
    logger.info('Loaded GloVe!')

    embeddings = np.asarray(embd, dtype=float)
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    return word_to_ix, embeddings


def glove_dump():
    filter_stop_words = False
    glove_file = args.glove
    out_file = args.output
    mention_files = [args.mentions]
    vocab = load_mentions_vocab_from_files(mention_files, filter_stop_words)
    word_to_ix, embeddings = load_glove_for_vocab(glove_file, vocab)

    logger.info('Words in vocabulary %d', len(vocab))
    logger.info('Found %d words from vocabulary', len(word_to_ix.keys()))
    with open(out_file, 'wb') as f:
        pickle.dump([word_to_ix, embeddings], f)
    logger.info('Saving dump to file-%s', out_file)


if __name__ == '__main__':
    io.validate_existing_filepath(args.mentions)
    io.validate_existing_filepath(args.output)
    io.validate_existing_filepath(args.glove)
    glove_dump()
