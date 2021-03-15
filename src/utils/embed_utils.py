import logging
import pickle
from typing import List

import torch
from transformers import RobertaTokenizer, RobertaModel

logger = logging.getLogger(__name__)


class EmbedTransformersGenerics(object):
    def __init__(self, max_surrounding_contx,
                 finetune=False, use_cuda=True):

        super(EmbedTransformersGenerics, self).__init__()
        self.model = RobertaModel.from_pretrained("roberta-large")
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        self.max_surrounding_contx = max_surrounding_contx
        self.use_cuda = use_cuda
        self.finetune = finetune
        self.embed_size = 1024

        if self.use_cuda:
            self.model.cuda()

    def get_mention_full_rep(self, mention):
        ment1_ids, ment1_inx_start, ment1_inx_end = self.mention_feat_to_vec(mention)

        if self.use_cuda:
            ment1_ids = ment1_ids.cuda()

        if not self.finetune:
            with torch.no_grad():
                last_hidden_span = self.model(ment1_ids, output_hidden_states=True).last_hidden_state[0]
        else:
            last_hidden_span = self.model(ment1_ids, output_hidden_states=True).last_hidden_state[0]

        last_hidden_span = last_hidden_span[ment1_inx_start:ment1_inx_end]
        return last_hidden_span, last_hidden_span[0], last_hidden_span[-1], last_hidden_span.shape[0]

    @staticmethod
    def extract_mention_surrounding_context(mention, max_surrounding_contx):
        tokens_inds = mention.tokens_number
        context = mention.mention_context
        start_mention_index = tokens_inds[0]
        end_mention_index = tokens_inds[-1] + 1

        if max_surrounding_contx != -1:
            context_before = start_mention_index - max_surrounding_contx
            context_after = end_mention_index + max_surrounding_contx
            if context_before < 0:
                context_before = 0

            if context_after > len(context):
                context_after = len(context)
        else:
            context_before = 0
            context_after = len(context)

        ret_context_before = context[context_before:start_mention_index]
        ret_mention = context[start_mention_index:end_mention_index]
        ret_context_after = context[end_mention_index:context_after]

        return ret_context_before, ret_mention, ret_context_after

    def mention_feat_to_vec(self, mention):
        cntx_before_str, ment_span_str, cntx_after_str = EmbedTransformersGenerics.\
            extract_mention_surrounding_context(mention, self.max_surrounding_contx)

        cntx_before, cntx_after = cntx_before_str, cntx_after_str
        try:
            if len(cntx_before_str) != 0:
                cntx_before = self.tokenizer.encode(" ".join(cntx_before_str), add_special_tokens=False)
            if len(cntx_after_str) != 0:
                cntx_after = self.tokenizer.encode(" ".join(cntx_after_str), add_special_tokens=False)
        except:
            print("FAILD on MentionID=" + mention.mention_id)
            raise

        if self.max_surrounding_contx != -1:
            if len(cntx_before) > self.max_surrounding_contx:
                cntx_before = [cntx_before[0]] + cntx_before[-self.max_surrounding_contx+1:]
            if len(cntx_after) > self.max_surrounding_contx:
                cntx_after = cntx_after[:self.max_surrounding_contx-1] + [cntx_after[-1]]

        ment_span = self.tokenizer.encode(" ".join(ment_span_str), add_special_tokens=False)

        if isinstance(ment_span, torch.Tensor):
            ment_span = ment_span.tolist()
        if isinstance(cntx_before, torch.Tensor):
            cntx_before = cntx_before.tolist()
        if isinstance(cntx_after, torch.Tensor):
            cntx_after = cntx_after.tolist()

        sent_tokens = torch.tensor([cntx_before + ment_span + cntx_after])
        return sent_tokens, len(cntx_before), len(cntx_before) + len(ment_span)

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
