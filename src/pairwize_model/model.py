import math
import torch

from torch import nn


class PairWiseModel(nn.Module):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, embed_utils, use_cuda=True):
        super(PairWiseModel, self).__init__()
        self.W = nn.Linear(f_hid_dim, f_out_dim)
        self.pairwize = PairWiseModel.get_sequential(f_in_dim, f_hid_dim)
        self.embed_utils = embed_utils
        self.use_cuda = use_cuda

    @staticmethod
    def get_sequential(ind, hidd):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(ind, hidd),
            nn.ReLU(),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
        )

    def forward(self, batch_features, bs):
        embeded_features, gold_labels = self.get_bert_rep(batch_features, bs)
        prediction = self.W(self.pairwize(embeded_features))
        return prediction, gold_labels

    def predict(self, batch_features, bs):
        output, gold_labels = self.__call__(batch_features, bs)
        # output = torch.softmax(output, dim=1)
        # _, prediction = torch.max(output, dim=1)
        prediction = torch.sigmoid(output)
        # prediction = torch.round(output)
        return prediction, gold_labels

    def get_bert_rep(self, batch_features, bs=None):
        # for mention1, mention2 in batch_features:
        mentions1, mentions2 = zip(*batch_features)
        # (1, 768), (1, 169)
        hiddens1 = self.embed_utils.get_mentions_mean_rep(mentions1)
        hiddens1, _ = zip(*hiddens1)
        hiddens1 = torch.cat(hiddens1)
        # (1, 768)
        hiddens2 = self.embed_utils.get_mentions_mean_rep(mentions2)
        hiddens2, _ = zip(*hiddens2)
        hiddens2 = torch.cat(hiddens2)

        # (32, 768), (32,768)
        span1_span2 = hiddens1 * hiddens2

        # 768 * 3 = (1, 2304)
        concat_result = torch.cat((hiddens1, hiddens2, span1_span2), dim=1)

        ret_golds = torch.tensor(self.get_gold_labels(batch_features))

        if self.use_cuda:
            concat_result = concat_result.cuda()
            ret_golds = ret_golds.cuda()

        return concat_result, ret_golds

    def get_gold_labels(self, batch_features):
        batch_labels = list()
        for mentions1, mentions2 in batch_features:
            gold_label = 1 if mentions1.coref_chain == mentions2.coref_chain else 0
            batch_labels.append(gold_label)
        return batch_labels

    def set_embed_utils(self, embed_utils):
        self.embed_utils = embed_utils


class PairWiseModelKenton(PairWiseModel):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, embed_utils, use_cuda):
        super(PairWiseModelKenton, self).__init__(f_in_dim, f_hid_dim, f_out_dim, embed_utils, use_cuda)
        self.attend = PairWiseModel.get_sequential(embed_utils.get_embed_size(), f_hid_dim)
        self.w_alpha = nn.Linear(f_hid_dim, 1)

    def get_bert_rep(self, batch_features, batch_size=32):
        mentions1, mentions2 = zip(*batch_features)
        # (batch_size, embed_utils.get_embed_size())
        hiddens1, first1_tok, last1_tok, ment1_size = zip(*self.embed_utils.get_mentions_rep(mentions1))
        # (batch_size, embed_utils.get_embed_size())
        hiddens2, first2_tok, last2_tok, ment2_size = zip(*self.embed_utils.get_mentions_rep(mentions2))

        max_ment_span = max([max(ment1_size), max(ment2_size)])
        hiddens1_pad = [torch.nn.functional.pad(hid, [0, 0, 0, max_ment_span - hid.shape[0]]) for hid in hiddens1]
        hiddens2_pad = [torch.nn.functional.pad(hid, [0, 0, 0, max_ment_span - hid.shape[0]]) for hid in hiddens2]

        hiddens1_pad = torch.cat(hiddens1_pad)
        hiddens2_pad = torch.cat(hiddens2_pad)
        first1_tok = torch.cat(first1_tok).reshape(batch_size, -1)
        first2_tok = torch.cat(first2_tok).reshape(batch_size, -1)
        last1_tok = torch.cat(last1_tok).reshape(batch_size, -1)
        last2_tok = torch.cat(last2_tok).reshape(batch_size, -1)

        if self.use_cuda:
            hiddens1_pad = hiddens1_pad.cuda()
            hiddens2_pad = hiddens2_pad.cuda()
            first1_tok = first1_tok.cuda()
            first2_tok = first2_tok.cuda()
            last1_tok = last1_tok.cuda()
            last2_tok = last2_tok.cuda()

        attend1 = self.attend(hiddens1_pad)
        attend2 = self.attend(hiddens2_pad)

        att1_w = self.w_alpha(attend1)
        att2_w = self.w_alpha(attend2)

        # Clean attention on padded tokens
        att1_w = att1_w.reshape(batch_size, max_ment_span)
        att2_w = att2_w.reshape(batch_size, max_ment_span)
        self.clean_attnd_on_zero(att1_w, ment1_size, att2_w, ment2_size, max_ment_span)

        att1_soft = torch.softmax(att1_w, dim=1)
        att2_soft = torch.softmax(att2_w, dim=1)
        hidden1_reshape = hiddens1_pad.reshape(batch_size, max_ment_span, -1)
        hidden2_reshape = hiddens2_pad.reshape(batch_size, max_ment_span, -1)
        att1_head = hidden1_reshape * att1_soft.reshape(batch_size, max_ment_span, 1)
        att2_head = hidden2_reshape * att2_soft.reshape(batch_size, max_ment_span, 1)

        g1 = torch.cat((first1_tok, last1_tok, torch.sum(att1_head, dim=1)), dim=1)
        g2 = torch.cat((first2_tok, last2_tok, torch.sum(att2_head, dim=1)), dim=1)

        span1_span2 = g1 * g2

        concat_result = torch.cat((g1, g2, span1_span2), dim=1)

        ret_golds = torch.tensor(self.get_gold_labels(batch_features))

        if self.use_cuda:
            concat_result = concat_result.cuda()
            ret_golds = ret_golds.cuda()

        return concat_result, ret_golds

    def clean_attnd_on_zero(self, attend1, ment_size1, attend2, ment_size2, max_mention_span):
        for i, vals in enumerate(list(zip(ment_size1, ment_size2))):
            val1, val2 = vals
            if val1 > max_mention_span or val2 > max_mention_span:
                raise Exception("Mention size exceed maximum!")

            attend1_fx = attend1[i:i + 1, 0:val1]
            attend1_fx = torch.nn.functional.pad(attend1_fx, [0, max_mention_span - val1, 0, 0], value=-math.inf)
            attend1[i:i + 1] = attend1_fx

            attend2_fx = attend2[i:i + 1, 0:val2]
            attend2_fx = torch.nn.functional.pad(attend2_fx, [0, max_mention_span - val2, 0, 0], value=-math.inf)
            attend2[i:i + 1] = attend2_fx
