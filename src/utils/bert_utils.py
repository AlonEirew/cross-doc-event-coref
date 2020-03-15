import logging
import pickle

import torch
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification

logger = logging.getLogger(__name__)


MAX_MENTION_SPAN = 7


class BertPretrainedUtils(nn.Module):
    def __init__(self, max_surrounding_contx=10, finetune=False, use_cuda=True, pad=False):
        super(BertPretrainedUtils, self).__init__()
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self._bert = BertForSequenceClassification.from_pretrained('bert-base-cased', output_hidden_states=True, output_attentions=True)

        self.max_surrounding_contx = max_surrounding_contx
        self.finetune = finetune
        self.pad = pad
        self.use_cuda = use_cuda

        if self.use_cuda:
            self._bert.cuda()

    def get_mention_mean_rep(self, mention):
        ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = self.mention_feat_to_vec(mention)
        with torch.no_grad():
            if self.use_cuda:
                ment1_ids = ment1_ids.cuda()

            all_hidden_states, all_attention_states = self._bert(ment1_ids)[-2:]

        # last_attend2 = all_attentions2[0]
        last_hidden_span = all_hidden_states[0].view(all_hidden_states[0].shape[1], -1)[
                            ment1_inx_start:ment1_inx_end]

        last_attention_span = all_attention_states[0].view(all_attention_states[0].shape[1], -1)[
                            ment1_inx_start:ment1_inx_end]

        span_hidden_last_mean = torch.mean(last_hidden_span, dim=0).reshape(1, -1)
        span_attend_last_mean = torch.mean(last_attention_span, dim=0).reshape(1, -1)

        return span_hidden_last_mean, span_attend_last_mean

    def get_mentions_rep(self, mentions_list):
        # embed_list = list()
        # for mention in mentions_list:
        #     embed_list.append(self.get_mention_full_rep(mention))

        ids, masks, starts, ends = ([], [], [], [])
        for mention in mentions_list:
            ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = self.mention_feat_to_vec(mention)
            ids.append(ment1_ids)
            masks.append(att_mask)
            starts.append(ment1_inx_start)
            ends.append(ment1_inx_end)

        ids = torch.stack(ids).reshape(len(mentions_list), -1)
        masks = torch.stack(masks).reshape(len(mentions_list), -1)

        if self.use_cuda:
            ids = ids.cuda()
            masks = masks.cuda()

        if not self.finetune:
            with torch.no_grad():
                all_hidden_states, _ = self._bert(ids, attention_mask=masks)[-2:]
        else:
            all_hidden_states, _ = self._bert(ids, attention_mask=masks)[-2:]

        # last_attend2 = all_attentions2[0]
        all_last_span_pad, all_first, all_last, all_len = ([], [], [], [])
        for i in range(len(mentions_list)):
            last_hidden_span = all_hidden_states[0][i].view(all_hidden_states[0][i].shape[0], -1)[
                               starts[i]:ends[i]]

            # Padding of mention (longest mentions is 7 tokens)
            last_hidden_span_pad = torch.nn.functional.pad(last_hidden_span, [0, 0, 0, MAX_MENTION_SPAN - last_hidden_span.shape[0]])
            all_last_span_pad.append(last_hidden_span_pad)
            all_first.append(last_hidden_span[0])
            all_last.append(last_hidden_span[-1])
            all_len.append(last_hidden_span.shape[0])

        return zip(all_last_span_pad, all_first, all_last, all_len)

    def get_mention_full_rep(self, mention):
        ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = self.mention_feat_to_vec(mention)

        if self.use_cuda:
            ment1_ids = ment1_ids.cuda()
            att_mask = att_mask.cuda()

        if not self.finetune:
            with torch.no_grad():
                all_hidden_states, _ = self._bert(ment1_ids, attention_mask=att_mask)[-2:]
        else:
            all_hidden_states, _ = self._bert(ment1_ids, attention_mask=att_mask)[-2:]

        # last_attend2 = all_attentions2[0]
        last_hidden_span = all_hidden_states[0].view(all_hidden_states[0].shape[1], -1)[
                           ment1_inx_start:ment1_inx_end]

        last_hidden_span_pad = torch.nn.functional.pad(last_hidden_span, [0, 0, 0, 7 - last_hidden_span.shape[0]])
        return last_hidden_span_pad, last_hidden_span[0], last_hidden_span[-1], last_hidden_span.shape[0]

    def mention_feat_to_vec(self, mention):
        cntx_before_str, ment_span_str, cntx_after_str = self.extract_mention_surrounding_context(mention)
        cntx_before, cntx_after = cntx_before_str, cntx_after_str
        try:
            if len(cntx_before_str) != 0:
                cntx_before = self._tokenizer.encode(cntx_before_str)[0:-1]
            if len(cntx_after_str) != 0:
                cntx_after = self._tokenizer.encode(cntx_after_str)[1:]
        except:
            print("FAILD on MentionID=" + mention.mention_id)
            raise

        ment_span = self._tokenizer.encode(ment_span_str)[1:-1]
        tokens_length = len(cntx_before) + len(cntx_after) + len(ment_span)
        att_mask = [1] * tokens_length
        if self.pad:
            padding_length = 75 - tokens_length
            padding = [0] * padding_length
            sent_tokens = cntx_before + ment_span + cntx_after + padding
            att_mask += padding
        else:
            sent_tokens = cntx_before + ment_span + cntx_after

        sent_tokens = torch.tensor([sent_tokens])
        att_mask = torch.tensor([att_mask])
        return sent_tokens, att_mask, len(cntx_before), len(cntx_before) + len(ment_span)

    def extract_mention_surrounding_context(self, mention):
        tokens_inds = mention.tokens_number
        context = mention.mention_context
        start_mention_id = tokens_inds[0]
        end_mention_id = tokens_inds[-1] + 1

        if self.max_surrounding_contx != -1:
            context_before = start_mention_id - self.max_surrounding_contx
            context_after = end_mention_id + self.max_surrounding_contx
            if context_before < 0:
                context_before = 0

            if context_after > len(context):
                context_after = len(context)
        else:
            context_before = 0
            context_after = len(context)

        ret_context_before = context[context_before:start_mention_id]
        ret_mention = context[start_mention_id:end_mention_id]
        ret_context_after = context[end_mention_id:context_after]

        return ret_context_before, ret_mention, ret_context_after


class BertFromFile(object):
    def __init__(self, files_to_load: list):
        bert_dict = dict()

        if files_to_load is not None and len(files_to_load) > 0:
            for file_ in files_to_load:
                bert_dict.update(pickle.load(open(file_, "rb")))
                logger.info("Bert representation loaded-" + file_)

        self.embeddings = list(bert_dict.values())
        self.embed_key = {k: i for i, k in enumerate(bert_dict.keys())}

    def get_mention_mean_rep(self, mention):
        return self.embeddings[self.embed_key[mention.mention_id]]

    def get_mention_full_rep(self, mention):
        return self.embeddings[self.embed_key[mention.mention_id]]

    def get_mentions_rep(self, mentions_list):
        embed_list = list()
        for mention in mentions_list:
            embed_list.append(self.embeddings[self.embed_key[mention.mention_id]])

        return embed_list
