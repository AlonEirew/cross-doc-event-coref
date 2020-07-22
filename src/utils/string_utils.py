import collections

import os
import re
import string
from typing import List

import spacy
from spacy.tokens import Doc

from src.utils.io_utils import load_json_file

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

STOP_WORDS_FILE = os.path.join(CURRENT_DIR, 'resources/stop_words_en.json')
DISAMBIGUATION_CATEGORY = ['disambig', 'disambiguation']


class StringUtils(object):
    spacy_parser = spacy.load('en_core_web_sm')
    spacy_no_parser = spacy.load('en_core_web_sm', disable=['parser'])
    spacy_no_tok = spacy.load('en_core_web_sm', disable=['tokenizer'])
    stop_words = []
    pronouns = []
    preposition = []
    determiners = []

    def __init__(self):
        pass

    @staticmethod
    def is_stop(token: str) -> bool:
        if not StringUtils.stop_words:
            StringUtils.stop_words = load_json_file(STOP_WORDS_FILE)
            StringUtils.stop_words.extend(DISAMBIGUATION_CATEGORY)
        if token not in StringUtils.stop_words:
            return False
        return True

    @staticmethod
    def find_head_lemma_pos_ner(x: str):
        """"

        :param x: mention
        :return: the head word and the head word lemma of the mention
        """
        head = "UNK"
        lemma = "UNK"
        pos = "UNK"
        ner = "UNK"

        # pylint: disable=not-callable
        doc = StringUtils.spacy_parser(x)
        for tok in doc:
            if tok.head == tok:
                head = tok.text
                lemma = tok.lemma_
                pos = tok.pos_

        for ent in doc.ents:
            if ent.root.text == head:
                ner = ent.label_

        return head, lemma, pos, ner

    @staticmethod
    def get_tokenized_string(not_tokenized_str):
        tokenized_str = list()
        doc = StringUtils.spacy_parser(not_tokenized_str)
        for sentence in doc.sents:
            sent_toks = list()
            for token in sentence:
                sent_toks.append((token.text, token.i))
            tokenized_str.append(sent_toks)

        return tokenized_str

    @staticmethod
    def is_verb_phrase(text):
        doc = StringUtils.spacy_parser(text)
        for tok in doc:
            if tok.pos_ == "VERB":
                return True
        return False

