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
                 mention_head_lemma: str, coref_chain: str, mention_type: str = 'NA', coref_link: str = "NA",
                 predicted_coref_chain: str = None, mention_pos: str = None,
                 mention_ner: str = None, mention_index: int = -1, gen_lemma: bool = False) -> None:
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
            predicted_coref_chain: str (should be field while evaluated)
            mention_pos: str
            mention_ner: str
            mention_index: in case order is of value (default = -1)
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
        self.predicted_coref_chain = predicted_coref_chain
        self.coref_link = coref_link

        if mention_id is None:
            self.mention_id = self.gen_mention_id()
        else:
            self.mention_id = str(mention_id)

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
            mention_type = None
            coref_link = "NA"
            predicted_coref_chain = None
            mention_context = None
            mention_pos = None
            mention_ner = None
            mention_index = -1

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

            if 'predicted_coref_chain' in mention_line:
                predicted_coref_chain = mention_line['predicted_coref_chain']

            if 'mention_index' in mention_line:
                mention_index = mention_line['mention_index']

            if 'coref_link' in mention_line:
                coref_link = mention_line['coref_link']

            mention_data = MentionData(mention_id,
                                       topic_id,
                                       doc_id,
                                       sent_id,
                                       tokens_numbers,
                                       mention_text,
                                       mention_context,
                                       mention_head=mention_head,
                                       mention_head_lemma=mention_head_lemma,
                                       coref_chain=coref_chain,
                                       mention_type=mention_type,
                                       coref_link=coref_link,
                                       predicted_coref_chain=predicted_coref_chain,
                                       mention_pos=mention_pos,
                                       mention_ner=mention_ner,
                                       mention_index=mention_index)
        except Exception:
            print('Unexpected error:', sys.exc_info()[0])
            raise Exception('failed reading json line-' + str(mention_line))

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
        if not mentions_json_file:
            return None

        all_mentions_only = load_json_file(mentions_json_file)
        running_index = 1
        mentions = []
        for mention_line in all_mentions_only:
            mention_data = MentionData.read_json_mention_data_line(mention_line)
            mention_data.mention_index = running_index
            mentions.append(mention_data)
            running_index += 1

        return mentions
