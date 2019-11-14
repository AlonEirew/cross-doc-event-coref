import json
import logging
import sys
from typing import List

from src.utils.io_utils import load_json_file
from src.utils.string_utils import StringUtils

logger = logging.getLogger(__name__)


class MentionDataLight(object):
    def __init__(self, tokens_str: str, mention_context: List[str] = None, mention_head: str = None,
                 mention_head_lemma: str = None, mention_pos: str = None,
                 mention_ner: str = None, gen_lemma: bool = False):
        """
        Object represent a mention with only text values
        Args:
            tokens_str: str the tokens combine text (join with space)
            mention_head: str
            mention_head_lemma: str
        """
        self.tokens_str = tokens_str
        self.mention_context = mention_context
        if not mention_head and not mention_head_lemma:
            if gen_lemma:
                self.mention_head, self.mention_head_lemma, self.mention_head_pos, \
                    self.mention_ner = StringUtils.find_head_lemma_pos_ner(str(tokens_str))
        else:
            self.mention_head = mention_head
            self.mention_head_lemma = mention_head_lemma
            self.mention_head_pos = mention_pos
            self.mention_ner = mention_ner


class MentionData(MentionDataLight):
    def __init__(self, mention_id, topic_id: str, doc_id: str, sent_id: int, tokens_numbers: List[int],
                 tokens_str: str, mention_context: List[str], mention_head: str,
                 mention_head_lemma: str, coref_chain: str, mention_type: str = 'NA',
                 is_continuous: bool = True, is_singleton: bool = False, score: float = float(-1),
                 predicted_coref_chain: str = None, mention_pos: str = None,
                 mention_ner: str = None, mention_index: int = -1, gen_lemma: bool = False,
                 min_span_str: str = None, min_span_ids: List[int] = None) -> None:
        """
        Object represent a mention

        Args:
            topic_id: str topic ID
            doc_id: str document ID
            sent_id: int sentence number
            tokens_numbers: List[int] - tokens numbers
            mention_context: List[str] - list of tokens strings
            coref_chain: str
            mention_type: str one of (HUM/NON/TIM/LOC/ACT/NEG)
            is_continuous: bool
            is_singleton: bool
            score: float
            predicted_coref_chain: str (should be field while evaluated)
            mention_pos: str
            mention_ner: str
            mention_index: in case order is of value (default = -1)
            min_span_str: the minimum span of the mention (extracted "Using Automatically
                    Extracted Minimum Spans to Disentangle Coreference Evaluation from Boundary Detection")
        """
        super(MentionData, self).__init__(tokens_str, mention_context, mention_head,
                                          mention_head_lemma, mention_pos,
                                          mention_ner, gen_lemma)
        self.topic_id = topic_id
        self.doc_id = doc_id
        self.sent_id = sent_id
        self.tokens_number = tokens_numbers
        self.mention_type = mention_type
        self.coref_chain = coref_chain
        self.is_continuous = is_continuous
        self.is_singleton = is_singleton
        self.score = score
        self.predicted_coref_chain = predicted_coref_chain
        self.mention_id = str(mention_id)
        self.min_span_str = min_span_str
        self.min_span_ids = min_span_ids

        if self.mention_id is None:
            self.mention_id = self.gen_mention_id()

        self.mention_index = mention_index

    @staticmethod
    def read_json_mention_data_line(mention_line):
        """
        Args:
            mention_line: a Json representation of a single mention

        Returns:
            MentionData object
        """
        # pylint: disable=too-many-branches

        try:
            mention_id = None
            topic_id = None
            coref_chain = None
            doc_id = None
            sent_id = None
            tokens_numbers = None
            score = -1
            mention_type = None
            predicted_coref_chain = None
            mention_context = None
            is_continue = False
            is_singleton = False
            mention_pos = None
            mention_ner = None
            mention_index = -1
            min_span_str = None
            min_span_ids = None

            mention_text = mention_line['tokens_str']

            if 'mention_id' in mention_line:
                mention_id = mention_line['mention_id']

            if 'topic_id' in mention_line:
                topic_id = mention_line['topic_id']

            if 'coref_chain' in mention_line:
                coref_chain = mention_line['coref_chain']

            if 'doc_id' in mention_line:
                doc_id = mention_line['doc_id']
                # if '.xml' not in doc_id:
                #     doc_id = doc_id + '.xml'

            if 'sent_id' in mention_line:
                sent_id = mention_line['sent_id']

            if 'tokens_number' in mention_line:
                tokens_numbers = mention_line['tokens_number']

            if 'mention_context' in mention_line:
                mention_context = mention_line['mention_context']

            if 'mention_head' in mention_line and 'mention_head_lemma' in mention_line:
                mention_head = mention_line['mention_head']
                mention_head_lemma = mention_line['mention_head_lemma']
                if 'mention_head_pos' in mention_line:
                    mention_pos = mention_line['mention_head_pos']
                if 'mention_ner' in mention_line:
                    mention_ner = mention_line['mention_ner']
            else:
                mention_head, mention_head_lemma, mention_pos, \
                    mention_ner = StringUtils.find_head_lemma_pos_ner(str(mention_text))

            if 'mention_type' in mention_line:
                mention_type = mention_line['mention_type']
            if 'score' in mention_line:
                score = mention_line['score']

            if 'is_continuous' in mention_line:
                is_continue = mention_line['is_continuous']

            if 'is_singleton' in mention_line:
                is_singleton = mention_line['is_singleton']

            if 'predicted_coref_chain' in mention_line:
                predicted_coref_chain = mention_line['predicted_coref_chain']

            if 'mention_index' in mention_line:
                mention_index = mention_line['mention_index']

            if 'min_span_str' in mention_line:
                min_span_str = mention_line['min_span_str']

            if 'min_span_ids' in mention_line:
                min_span_ids = mention_line['min_span_ids']

            mention_data = MentionData(mention_id, topic_id, doc_id, sent_id, tokens_numbers, mention_text,
                                       mention_context,
                                       mention_head, mention_head_lemma,
                                       coref_chain, mention_type, is_continue, is_singleton, score,
                                       predicted_coref_chain, mention_pos, mention_ner,
                                       mention_index, min_span_str=min_span_str, min_span_ids=min_span_ids)
        except Exception:
            print('Unexpected error:', sys.exc_info()[0])
            raise Exception('failed reading json line-' + str(mention_line))

        return mention_data

    @staticmethod
    def read_sqlite_mention_data_line_v8(mention_line, gen_lemma=True, extract_valid_sent=False):
        coref_chain = mention_line[0]
        mention_text = mention_line[1]
        token_start = mention_line[2]
        token_end = mention_line[3]
        doc_id = mention_line[4]
        mention_context = mention_line[5]
        mention_type = mention_line[9]
        mention_id = mention_line[10]

        if extract_valid_sent:
            accum_sent_start = 0
            # if mention_text == 'following year':
            tokenized_context = StringUtils.get_tokenized_string(mention_context)
            # split_sentences = mention_context.split('.')
            for sentence in tokenized_context:
                # split_sentence = sentence.strip().split(' ')
                sent_start = accum_sent_start
                sent_end = sent_start + len(sentence)
                if sent_start <= token_start <= sent_end:
                    token_start = token_start - sent_start
                    token_end = token_end - sent_start
                    mention_context = ' '.join([tup[0] for tup in sentence])
                    break
                else:
                    accum_sent_start = sent_end

        tokens_numbers = list()
        for i in range(token_start, token_end):
            tokens_numbers.append(i)
        tokens_numbers.append(token_end)

        mention_data = MentionData(mention_id, -1, doc_id, -1, tokens_numbers, mention_text,
                                   mention_context.split(' '), None, None, coref_chain, mention_type, gen_lemma=gen_lemma)

        mention_data.mention_id = mention_id
        return mention_data

    @staticmethod
    def read_sqlite_mention_data_line_v9(mention_line, gen_lemma=True, extract_valid_sent=True):
        mention_data = MentionData.read_sqlite_mention_data_line_v8(mention_line, gen_lemma, False)
        token_start = mention_line[2]
        token_end = mention_line[3]
        mention_context = mention_line[5]
        mention_data.tokens_number = list()

        context_json = json.loads(mention_context)
        final_context = list()
        if context_json:
            sent_id = 0
            for sentence in context_json:
                for token_dict in sentence:
                    token_str, token_index = list(token_dict.items())[0]
                    final_context.append(token_str)
                    if token_start <= token_index <= token_end:
                        mention_data.tokens_number.append(token_index)
                        mention_data.sent_id = sent_id

                sent_id += 1

        if extract_valid_sent:
            tmp_context = context_json[mention_data.sent_id]
            sentence_start_token = tmp_context[0]
            sentence_start_token_str, sentence_start_token_index = list(sentence_start_token.items())[0]
            for i in range(len(mention_data.tokens_number)):
                mention_data.tokens_number[i] = mention_data.tokens_number[i] - sentence_start_token_index

            final_context = list()
            for sentence_tokens in tmp_context:
                token_str, token_index = list(sentence_tokens.items())[0]
                final_context.append(token_str)

        mention_data.mention_context = final_context

        return mention_data

    def get_tokens(self):
        return self.tokens_number

    def gen_mention_id(self) -> str:
        if self.doc_id and self.sent_id is not None and self.tokens_number:
            tokens_ids = [str(self.doc_id), str(self.sent_id)]
            tokens_ids.extend([str(token_id) for token_id in self.tokens_number])
            return '_'.join(tokens_ids)

        return '_'.join(self.tokens_str.split())

    def get_mention_id(self) -> str:
        if not self.mention_id:
            self.mention_id = self.gen_mention_id()
        return self.mention_id

    @staticmethod
    def static_gen_token_unique_id(doc_id: int, sent_id: int, token_id: int) -> str:
        return '_'.join([str(doc_id), str(sent_id), str(token_id)])

    @staticmethod
    def load_mentions_vocab_from_files(mentions_files, filter_stop_words=False):
        logger.info('Loading mentions files...')
        mentions = []
        for _file in mentions_files:
            mentions.extend(MentionData.read_mentions_json_to_mentions_data_list(_file))

        return MentionData.load_mentions_vocab(mentions, filter_stop_words)

    @staticmethod
    def load_mentions_vocab(mentions, filter_stop_words=False):
        vocab = MentionData.extract_vocab(mentions, filter_stop_words)
        logger.info('Done loading mentions files...')
        return vocab

    @staticmethod
    def extract_vocab(mentions, filter_stop_words: bool) -> List[str]:
        """
        Extract Head, Lemma and mention string from all mentions to create a list of string vocabulary
        Args:
            mentions:
            filter_stop_words:

        Returns:

        """
        vocab = set()
        for mention in mentions:
            head = mention.mention_head
            head_lemma = mention.mention_head_lemma
            tokens_str = mention.tokens_str
            if not filter_stop_words:
                vocab.add(head)
                vocab.add(head_lemma)
                vocab.add(tokens_str)
            else:
                if not StringUtils.is_stop(head):
                    vocab.add(head)
                if not StringUtils.is_stop(head_lemma):
                    vocab.add(head_lemma)
                if not StringUtils.is_stop(tokens_str):
                    vocab.add(tokens_str)
        vocab_set = list(vocab)
        return vocab_set

    @staticmethod
    def read_mentions_json_to_mentions_data_list(mentions_json_file: str):
        """

        Args:
            mentions_json_file: the path of the mentions json file to read

        Returns:
            List[MentionData]
        """
        all_mentions_only = load_json_file(mentions_json_file)

        mentions = []
        for mention_line in all_mentions_only:
            mention_data = MentionData.read_json_mention_data_line(mention_line)
            mentions.append(mention_data)

        return mentions
