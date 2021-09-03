import logging
import pickle
from typing import List

import torch
from transformers import RobertaTokenizer, RobertaModel, BertModel, BertTokenizer

logger = logging.getLogger(__name__)


class EmbedTransformersGenerics(object):
    def __init__(self, max_surrounding_contx,
                 finetune=False, use_cuda=True):

        self.model = RobertaModel.from_pretrained("roberta-large")
        # self.model = BertModel.from_pretrained("bert-large-cased")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        # self.tokenizer = BertTokenizer.from_pretrained("bert-large-cased")
        self.max_surrounding_contx = max_surrounding_contx
        self.use_cuda = use_cuda
        self.finetune = finetune
        self.embed_size = 1024

        if self.use_cuda:
            self.model.cuda()

    def get_mention_full_rep(self, mention):
        sent_ids, ment1_inx_start, ment1_inx_end = self.mention_feat_to_vec(mention)

        if self.use_cuda:
            sent_ids = sent_ids.cuda()

        if not self.finetune:
            with torch.no_grad():
                last_hidden_span = self.model(sent_ids).last_hidden_state
        else:
            last_hidden_span = self.model(sent_ids).last_hidden_state

        mention_hidden_span = last_hidden_span.view(last_hidden_span.shape[1], -1)[ment1_inx_start:ment1_inx_end]
        return mention_hidden_span, mention_hidden_span[0], mention_hidden_span[-1], mention_hidden_span.shape[0]

    @staticmethod
    def extract_mention_surrounding_context(mention):
        tokens_inds = mention.tokens_number
        context = mention.mention_context
        start_mention_index = tokens_inds[0]
        end_mention_index = tokens_inds[-1] + 1
        assert len(tokens_inds) == len(mention.tokens_str.split(" "))

        ret_context_before = context[0:start_mention_index]
        ret_mention = context[start_mention_index:end_mention_index]
        ret_context_after = context[end_mention_index:]

        assert ret_mention == mention.tokens_str.split(" ")
        assert ret_context_before + ret_mention + ret_context_after == mention.mention_context

        return ret_context_before, ret_mention, ret_context_after

    def mention_feat_to_vec(self, mention):
        cntx_before_str, ment_span_str, cntx_after_str = EmbedTransformersGenerics.\
            extract_mention_surrounding_context(mention)

        cntx_before, cntx_after = cntx_before_str, cntx_after_str
        if len(cntx_before_str) != 0:
            cntx_before = self.tokenizer.encode(" ".join(cntx_before_str), add_special_tokens=False)
        if len(cntx_after_str) != 0:
            cntx_after = self.tokenizer.encode(" ".join(cntx_after_str), add_special_tokens=False)

        if self.max_surrounding_contx != -1:
            if len(cntx_before) > self.max_surrounding_contx:
                cntx_before = cntx_before[-self.max_surrounding_contx+1:]
            if len(cntx_after) > self.max_surrounding_contx:
                cntx_after = cntx_after[:self.max_surrounding_contx-1]

        ment_span = self.tokenizer.encode(" ".join(ment_span_str), add_special_tokens=False)

        if isinstance(ment_span, torch.Tensor):
            ment_span = ment_span.tolist()
        if isinstance(cntx_before, torch.Tensor):
            cntx_before = cntx_before.tolist()
        if isinstance(cntx_after, torch.Tensor):
            cntx_after = cntx_after.tolist()

        all_sent_toks = [[0] + cntx_before + ment_span + cntx_after + [2]]
        sent_tokens = torch.tensor(all_sent_toks)
        mention_start_idx = len(cntx_before) + 1
        mention_end_idx = len(cntx_before) + len(ment_span) + 1
        assert all_sent_toks[0][mention_start_idx:mention_end_idx] == ment_span
        return sent_tokens, mention_start_idx, mention_end_idx

    def get_embed_size(self):
        return self.embed_size


class EmbedFromFile(object):
    def __init__(self, files_to_load: List[str]):
        """
        :param files_to_load: list of pre-generated embedding file names
        """
        self.embed_size = 1024
        bert_dict = dict()

        if files_to_load is not None and len(files_to_load) > 0:
            for file_ in files_to_load:
                bert_dict.update(pickle.load(open(file_, "rb")))
                logger.info("Bert representation loaded-" + file_)

        self.embeddings = list(bert_dict.values())
        self.embed_key = {k: i for i, k in enumerate(bert_dict.keys())}

    def get_mention_full_rep(self, mention):
        return self.embeddings[self.embed_key[mention.mention_id]]

    def get_mentions_rep(self, mentions_list):
        embed_list = [self.embeddings[self.embed_key[mention.mention_id]] for mention in mentions_list]
        return embed_list

    def get_embed_size(self):
        return self.embed_size
