import math
import torch

from torch import nn


class PairWiseModel(nn.Module):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, bert_utils, use_cuda=True):
        super(PairWiseModel, self).__init__()
        self.W = nn.Linear(f_hid_dim, f_out_dim)
        self.pairwize = PairWiseModel.get_sequential(f_in_dim, f_hid_dim)
        self.bert_utils = bert_utils
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
        output = torch.softmax(output, dim=1)
        _, prediction = torch.max(output, dim=1)
        # output = torch.sigmoid(output)
        # prediction = torch.round(output)
        return prediction, gold_labels

    def get_bert_rep(self, batch_features, bs=None):
        # for mention1, mention2 in batch_features:
        mentions1, mentions2 = zip(*batch_features)
        # (1, 768), (1, 169)
        hiddens1 = self.bert_utils.get_mentions_mean_rep(mentions1)
        hiddens1, _ = zip(*hiddens1)
        hiddens1 = torch.cat(hiddens1)
        # (1, 768)
        hiddens2 = self.bert_utils.get_mentions_mean_rep(mentions2)
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


class PairWiseModelKenton(PairWiseModel):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim, bert_utils, use_cuda):
        super(PairWiseModelKenton, self).__init__(f_in_dim, f_hid_dim, f_out_dim, bert_utils, use_cuda)
        self.attend = PairWiseModel.get_sequential(5376, f_hid_dim)
        self.w_alpha = nn.Linear(f_hid_dim, 7)

    def get_bert_rep(self, batch_features, batch_size=32):
        mentions1, mentions2 = zip(*batch_features)
        # (x, 768)
        hiddens1, first1_tok, last1_tok, ment1_size = zip(*self.bert_utils.get_mentions_rep(mentions1))
        hiddens1 = torch.cat(hiddens1).reshape(batch_size, -1)
        first1_tok = torch.cat(first1_tok).reshape(batch_size, -1)
        last1_tok = torch.cat(last1_tok).reshape(batch_size, -1)
        # (x, 768)
        hiddens2, first2_tok, last2_tok, ment2_size = zip(*self.bert_utils.get_mentions_rep(mentions2))
        hiddens2 = torch.cat(hiddens2).reshape(batch_size, -1)
        first2_tok = torch.cat(first2_tok).reshape(batch_size, -1)
        last2_tok = torch.cat(last2_tok).reshape(batch_size, -1)

        if self.use_cuda:
            hiddens1 = hiddens1.cuda()
            hiddens2 = hiddens2.cuda()
            first1_tok = first1_tok.cuda()
            first2_tok = first2_tok.cuda()
            last1_tok = last1_tok.cuda()
            last2_tok = last2_tok.cuda()

        attend1 = self.attend(hiddens1)
        attend2 = self.attend(hiddens2)

        att1_w = self.w_alpha(attend1)
        att2_w = self.w_alpha(attend2)

        # Clean attention on padded tokens
        self.clean_attnd_on_zero(att1_w, ment1_size, att2_w, ment2_size)

        att1_soft = torch.softmax(att1_w, dim=1)
        att2_soft = torch.softmax(att2_w, dim=1)
        hidden1_reshape = hiddens1.reshape(batch_size, 7, -1)
        hidden2_reshape = hiddens2.reshape(batch_size, 7, -1)
        att1_head = hidden1_reshape * att1_soft.reshape(batch_size, 7, 1)
        att2_head = hidden2_reshape * att2_soft.reshape(batch_size, 7, 1)

        g1 = torch.cat((first1_tok, last1_tok, att1_head.reshape(batch_size, -1)), dim=1)
        g2 = torch.cat((first2_tok, last2_tok, att2_head.reshape(batch_size, -1)), dim=1)

        # (1, 6912), (1,6912)
        span1_span2 = g1 * g2

        # 6912 * 3 = (1, 20736)
        concat_result = torch.cat((g1, g2, span1_span2), dim=1)

        ret_golds = torch.tensor(self.get_gold_labels(batch_features))

        if self.use_cuda:
            concat_result = concat_result.cuda()
            ret_golds = ret_golds.cuda()

        return concat_result, ret_golds

    def clean_attnd_on_zero(self, attend1, ment_size1, attend2, ment_size2):
        for i, vals in enumerate(list(zip(ment_size1, ment_size2))):
            val1, val2 = vals
            if val1 > 7:
                val1 = 7

            if val2 > 7:
                val2 = 7

            attend1_fx = attend1[i:i + 1, 0:val1]
            attend1_fx = torch.nn.functional.pad(attend1_fx, [0, 7 - val1, 0, 0], value=-math.inf)
            attend1[i:i + 1] = attend1_fx

            attend2_fx = attend2[i:i + 1, 0:val2]
            attend2_fx = torch.nn.functional.pad(attend2_fx, [0, 7 - val2, 0, 0], value=-math.inf)
            attend2[i:i + 1] = attend2_fx
