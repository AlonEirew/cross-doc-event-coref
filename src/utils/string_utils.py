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
PRONOUN_FILE = os.path.join(CURRENT_DIR, 'resources/pronoun_en.json')
PREPOSITION_FILE = os.path.join(CURRENT_DIR, 'resources/preposition_en.json')
DETERMINERS_FILE = os.path.join(CURRENT_DIR, 'resources/determiners_en.json')

DISAMBIGUATION_CATEGORY = ['disambig', 'disambiguation']


class StringUtils(object):
    spacy_parser = spacy.load('en_core_web_sm')
    spacy_no_parser = spacy.load('en_core_web_sm', disable=['parser'])
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
    def normalize_str(in_str: str) -> str:
        str_clean = re.sub('[' + string.punctuation + string.whitespace + ']', ' ',
                           in_str).strip().lower()
        if isinstance(str_clean, str):
            str_clean = str(str_clean)

        doc = StringUtils.spacy_no_parser(str_clean)
        ret_clean = []
        for token in doc:
            lemma = token.lemma_.strip()
            if not StringUtils.is_pronoun(lemma) and not StringUtils.is_stop(lemma):
                ret_clean.append(token.lemma_)

        return ' '.join(ret_clean)

    @staticmethod
    def is_pronoun(in_str: str) -> bool:
        if not StringUtils.pronouns:
            StringUtils.pronouns = load_json_file(PRONOUN_FILE)

        tokens = in_str.split()
        if len(tokens) == 1:
            if tokens[0] in StringUtils.pronouns:
                return True
        return False

    @staticmethod
    def is_determiner(in_str: str) -> bool:
        if not StringUtils.determiners:
            StringUtils.determiners = load_json_file(DETERMINERS_FILE)

        tokens = in_str.split()
        if len(tokens) == 1:
            if tokens[0] in StringUtils.determiners:
                return True
        return False

    @staticmethod
    def is_preposition(in_str: str) -> bool:
        if not StringUtils.preposition:
            StringUtils.preposition = load_json_file(PREPOSITION_FILE)

        tokens = in_str.split()
        if len(tokens) == 1:
            if tokens[0] in StringUtils.preposition:
                return True
        return False

    @staticmethod
    def normalize_string_list(str_list: str) -> List[str]:
        ret_list = []
        for _str in str_list:
            normalize_str = StringUtils.normalize_str(_str)
            if normalize_str != '':
                ret_list.append(normalize_str)
        return ret_list

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
    def find_all_ners(x: List[str]):
        """

        :param x: context with mention
        :return: the list of all context tokens ners
        """
        ners = list()

        # pylint: disable=not-callable
        doc = Doc(StringUtils.spacy_parser.vocab, words=x, spaces=[True] * len(x))
        # parsed = StringUtils.spacy_parser(doc.text)
        for name, proc in StringUtils.spacy_parser.pipeline:  # iterate over components in order
            parsed = proc(doc)
            if name == 'ner':
                for tok in parsed:
                    ners.append(tok.ent_type_)

        return ners

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
    def get_tokens_from_list(in_token: List[str]):
        return StringUtils.spacy_parser.tokenizer.tokens_from_list(in_token)

    @staticmethod
    def is_verb_phrase(text):
        doc = StringUtils.spacy_parser(text)
        for tok in doc:
            if tok.pos_ == "VERB":
                return True
        return False
