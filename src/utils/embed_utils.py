import logging
import pickle

import torch
from enum import Enum
from torch import nn
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaModel

logger = logging.getLogger(__name__)


class EmbeddingEnum(Enum):
    BERT_LARGE_CASED = 1
    BERT_BASE_CASED = 2
    ROBERTA_LARGE = 3


class EmbeddingConfig(object):
    def __init__(self, embed_enum: EmbeddingEnum):
        if embed_enum == EmbeddingEnum.BERT_BASE_CASED:
            self.model_name = "bert-base-cased"
            self.model_size = 768
        elif embed_enum == EmbeddingEnum.BERT_LARGE_CASED:
            self.model_name = "bert-large-cased"
            self.model_size = 1024
        elif embed_enum == EmbeddingEnum.ROBERTA_LARGE:
            self.model_name = "roberta-large"
            self.model_size = 1024


class EmbedTransformersGenerics(nn.Module):
    def __init__(self, model=None, tokenizer=None, max_surrounding_contx=10,
                 max_mention_span=-1, finetune=False, use_cuda=True, pad=False,
                 embed_size=-1):

        super(EmbedTransformersGenerics, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_mention_span = max_mention_span
        self.max_surrounding_contx=max_surrounding_contx
        self.pad = pad
        self.use_cuda = use_cuda
        self.finetune = finetune
        self.embed_size = embed_size

        if self.use_cuda:
            self.model.cuda()

    def get_mention_full_rep(self, mention):
        raise NotImplementedError("Method implemented only in subclasses")

    @staticmethod
    def extract_mention_surrounding_context(mention, max_surrounding_contx):
        tokens_inds = mention.tokens_number
        context = mention.mention_context
        start_mention_id = tokens_inds[0]
        end_mention_id = tokens_inds[-1] + 1

        if max_surrounding_contx != -1:
            context_before = start_mention_id - max_surrounding_contx
            context_after = end_mention_id + max_surrounding_contx
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

    @staticmethod
    def mention_feat_to_vec(mention, tokenizer, max_surrounding_contx, pad):
        cntx_before_str, ment_span_str, cntx_after_str = EmbedTransformersGenerics.\
            extract_mention_surrounding_context(mention, max_surrounding_contx)

        cntx_before, cntx_after = cntx_before_str, cntx_after_str
        try:
            if len(cntx_before_str) != 0:
                cntx_before = tokenizer.encode(" ".join(cntx_before_str))[0:-1]
            if len(cntx_after_str) != 0:
                cntx_after = tokenizer.encode(" ".join(cntx_after_str))[1:]
        except:
            print("FAILD on MentionID=" + mention.mention_id)
            raise

        if len(cntx_before) > max_surrounding_contx:
            cntx_before = [cntx_before[0]] + cntx_before[-max_surrounding_contx+1:]
        if len(cntx_after) > max_surrounding_contx:
            cntx_after = cntx_after[:max_surrounding_contx-1] + [cntx_after[-1]]

        ment_span = tokenizer.encode(" ".join(ment_span_str))[1:-1]
        tokens_length = len(cntx_before) + len(cntx_after) + len(ment_span)
        att_mask = [1] * tokens_length

        if isinstance(ment_span, torch.Tensor):
            ment_span = ment_span.tolist()
        if isinstance(cntx_before, torch.Tensor):
            cntx_before = cntx_before.tolist()
        if isinstance(cntx_after, torch.Tensor):
            cntx_after = cntx_after.tolist()

        if pad:
            padding_length = 75 - tokens_length
            padding = [0] * padding_length
            sent_tokens = cntx_before + ment_span + cntx_after + padding
            att_mask += padding
        else:
            sent_tokens = cntx_before + ment_span + cntx_after

        sent_tokens = torch.tensor([sent_tokens])
        att_mask = torch.tensor([att_mask])
        return sent_tokens, att_mask, len(cntx_before), len(cntx_before) + len(ment_span)


class BertPretrainedUtils(EmbedTransformersGenerics):
    def __init__(self, bert_config: EmbeddingConfig, max_surrounding_contx=10, max_mention_span=-1,
                 finetune=False, use_cuda=True, pad=False):

        super(BertPretrainedUtils, self).__init__(model=BertForSequenceClassification.from_pretrained(bert_config.model_name, output_hidden_states=True, output_attentions=True),
                                                  tokenizer=BertTokenizer.from_pretrained(bert_config.model_name),
                                                  max_mention_span=max_mention_span,
                                                  max_surrounding_contx=max_surrounding_contx,
                                                  finetune=finetune, use_cuda=use_cuda, pad=pad,
                                                  embed_size=bert_config.model_size)

    def get_mention_mean_rep(self, mention):
        ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = EmbedTransformersGenerics.mention_feat_to_vec(
            mention, self.tokenizer, self.max_surrounding_contx, self.pad)
        with torch.no_grad():
            if self.use_cuda:
                ment1_ids = ment1_ids.cuda()

            all_hidden_states, all_attention_states = self.model(ment1_ids)[-2:]

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
            ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = EmbedTransformersGenerics.mention_feat_to_vec(
                mention, self.tokenizer, self.max_surrounding_contx, self.pad)

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
                all_hidden_states, _ = self.model(ids, attention_mask=masks)[-2:]
        else:
            all_hidden_states, _ = self.model(ids, attention_mask=masks)[-2:]

        # last_attend2 = all_attentions2[0]
        all_last_span_pad, all_first, all_last, all_len = ([], [], [], [])
        for i in range(len(mentions_list)):
            last_hidden_span = all_hidden_states[0][i].view(all_hidden_states[0][i].shape[0], -1)[
                               starts[i]:ends[i]]

            # Padding of mention (longest mentions is 7 tokens)
            last_hidden_span_pad = torch.nn.functional.pad(last_hidden_span, [0, 0, 0, self.max_mention_span - last_hidden_span.shape[0]])
            all_last_span_pad.append(last_hidden_span_pad)
            all_first.append(last_hidden_span[0])
            all_last.append(last_hidden_span[-1])
            all_len.append(last_hidden_span.shape[0])

        return zip(all_last_span_pad, all_first, all_last, all_len)

    def get_mention_full_rep(self, mention):
        ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = EmbedTransformersGenerics.mention_feat_to_vec(
            mention, self.tokenizer, self.max_surrounding_contx, self.pad)

        try:
            if self.use_cuda:
                ment1_ids = ment1_ids.cuda()
                att_mask = att_mask.cuda()

            if not self.finetune:
                with torch.no_grad():
                    all_hidden_states, _ = self.model(ment1_ids, attention_mask=att_mask)[-2:]
            else:
                all_hidden_states, _ = self.model(ment1_ids, attention_mask=att_mask)[-2:]
        except:
            print("FAILD on TopicId:" + mention.topic_id + " MentionID=" + mention.mention_id)
            raise

        # last_attend2 = all_attentions2[0]
        last_hidden_span = all_hidden_states[0].view(all_hidden_states[0].shape[1], -1)[
                           ment1_inx_start:ment1_inx_end]

        last_hidden_span_pad = torch.nn.functional.pad(last_hidden_span, [0, 0, 0, self.max_mention_span - last_hidden_span.shape[0]])
        return last_hidden_span_pad, last_hidden_span[0], last_hidden_span[-1], last_hidden_span.shape[0]

    def get_embed_size(self):
        return self.embed_size


class BertFromFile(object):
    def __init__(self, files_to_load: list, embed_size, max_mention_span):
        bert_dict = dict()
        self.embed_size = embed_size
        self.max_mention_span = max_mention_span

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
            ment_embed = self.embeddings[self.embed_key[mention.mention_id]]
            hidd = ment_embed[0][0:self.max_mention_span]
            ment_embed = (hidd, ment_embed[1], ment_embed[2], ment_embed[3])
            embed_list.append(ment_embed)

        return embed_list

    def get_embed_size(self):
        return self.embed_size


class RoBERTaPretrainedUtils(EmbedTransformersGenerics):
    def __init__(self, roberta_config: EmbeddingConfig,
                 max_surrounding_contx=10, max_mention_span=-1, finetune=False, use_cuda=True, pad=False):

        super(RoBERTaPretrainedUtils, self).__init__(model=RobertaModel.from_pretrained(roberta_config.model_name),
                                                     tokenizer=RobertaTokenizer.from_pretrained(roberta_config.model_name),
                                                     max_mention_span=max_mention_span,
                                                     max_surrounding_contx=max_surrounding_contx,
                                                     finetune=finetune, use_cuda=use_cuda, pad=pad,
                                                     embed_size=roberta_config.model_size)

    def get_mention_full_rep(self, mention):
        ment1_ids, att_mask, ment1_inx_start, ment1_inx_end = EmbedTransformersGenerics.mention_feat_to_vec(
            mention, self.tokenizer, self.max_surrounding_contx, self.pad)

        if self.use_cuda:
            ment1_ids = ment1_ids.cuda()

        if not self.finetune:
            with torch.no_grad():
                last_hidden_span, _ = self.model(ment1_ids)
        else:
            last_hidden_span, _ = self.model(ment1_ids)

        last_hidden_span = last_hidden_span.view(last_hidden_span.shape[1], -1)[
                           ment1_inx_start:ment1_inx_end]

        last_hidden_span_pad = torch.nn.functional.pad(last_hidden_span, [0, 0, 0, self.max_mention_span - last_hidden_span.shape[0]])
        return last_hidden_span_pad, last_hidden_span[0], last_hidden_span[-1], last_hidden_span.shape[0]

    def get_embed_size(self):
        return self.embed_size
