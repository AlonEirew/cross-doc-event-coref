import torch

from torch import nn


class PairWiseModel(nn.Module):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim):
        super(PairWiseModel, self).__init__()
        self.W = nn.Linear(f_hid_dim, f_out_dim)
        self.pairwize = PairWiseModel.get_sequential(f_in_dim, f_hid_dim)

    @staticmethod
    def get_sequential(ind, hidd):
        return nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(ind, hidd),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(hidd, hidd),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

    def forward(self, features):
        output = self.W(self.pairwize(features))
        return output

    def predict(self, features):
        output = self.__call__(features)
        output = torch.softmax(output, dim=1)
        _, prediction = torch.max(output, dim=1)
        # output = torch.sigmoid(output)
        # prediction = torch.round(output)
        return prediction

    def get_bert_rep(self, batch_features, bert_utils, use_cuda):
        batch_result = list()
        batch_labels = list()
        for mention1, mention2 in batch_features:
            # (1, 768), (1, 169)
            hidden1 = bert_utils.get_mention_mean_rep(mention1)
            if type(hidden1) == tuple:
                hidden1, _ = hidden1
            # (1, 768)
            hidden2 = bert_utils.get_mention_mean_rep(mention2)
            if type(hidden2) == tuple:
                hidden2, _ = hidden2

            # (1, 768), (1,169)
            span1_span2 = hidden1 * hidden2

            # 768 * 2 + 1 = (1, 2304)
            concat_result = torch.cat((hidden1.reshape(-1), hidden2.reshape(-1), span1_span2.reshape(-1))).reshape(1,
                                                                                                                   -1)
            gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0

            batch_result.append(concat_result)
            batch_labels.append(gold_label)

        ret_result = torch.cat(batch_result)
        ret_golds = torch.tensor(batch_labels)

        if use_cuda:
            ret_result = ret_result.cuda()
            ret_golds = ret_golds.cuda()

        return ret_result, ret_golds


class PairWiseModelKenton(PairWiseModel):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim):
        super(PairWiseModelKenton, self).__init__(f_in_dim, f_hid_dim, f_out_dim)
        self.attend = PairWiseModel.get_sequential(768, f_hid_dim)
        self.w_alpha = nn.Linear(f_hid_dim, 1)

    def get_bert_rep(self, batch_features, bert_utils, use_cuda):
        batch_result = list()
        batch_labels = list()
        for mention1, mention2 in batch_features:
            # (x, 768)
            hidden1 = bert_utils.get_mention_full_rep(mention1)
            if type(hidden1) == tuple:
                hidden1, _ = hidden1
            # (x, 768)
            hidden2 = bert_utils.get_mention_full_rep(mention2)
            if type(hidden2) == tuple:
                hidden2, _ = hidden2

            pad_hidden1 = torch.nn.functional.pad(hidden1, (0, 0, 0, 7 - hidden1.shape[0]))
            pad_hidden2 = torch.nn.functional.pad(hidden2, (0, 0, 0, 7 - hidden2.shape[0]))

            attend1 = self.attend(pad_hidden1)
            attend2 = self.attend(pad_hidden2)
            att1_w = self.w_alpha(attend1)
            att2_w = self.w_alpha(attend2)
            att1_soft = torch.softmax(att1_w, dim=0)
            att2_soft = torch.softmax(att2_w, dim=0)
            att1_head = pad_hidden1 * att1_soft
            att2_head = pad_hidden2 * att2_soft

            g1 = torch.cat((hidden1[0], hidden1[-1], att1_head.reshape(-1)))
            g2 = torch.cat((hidden2[0], hidden2[-1], att2_head.reshape(-1)))

            # (1, 6912), (1,6912)
            span1_span2 = g1 * g2

            # 768 * 2 + 1 = (1, 2304)
            concat_result = torch.cat((g1, g2, span1_span2.reshape(-1))).reshape(1, -1)

            gold_label = 1 if mention1.coref_chain == mention2.coref_chain else 0

            batch_result.append(concat_result)
            batch_labels.append(gold_label)

        ret_result = torch.cat(batch_result)
        ret_golds = torch.tensor(batch_labels)

        if use_cuda:
            ret_result = ret_result.cuda()
            ret_golds = ret_golds.cuda()

        return ret_result, ret_golds
