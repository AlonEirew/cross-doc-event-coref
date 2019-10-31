import torch
import torch.nn.functional as F

from torch import nn


class SnliCorefModel(nn.Module):
    def __init__(self, f_in_dim):
        super(SnliCorefModel, self).__init__()

        self.project_embedd = None

        g_in = 2 * f_in_dim
        g_hid = g_in // 16
        g_out = g_hid // 2

        h_in = 2 * g_out
        h_hid = h_in // 2
        h_out = h_hid // 2

        self.F = SnliCorefModel.get_sequential(f_in_dim, f_in_dim // 8, f_in_dim // 16)
        self.G = SnliCorefModel.get_sequential(g_in, g_hid, g_out)
        self.H = SnliCorefModel.get_sequential(h_in, h_hid, h_out)
        self.last_layer = nn.Linear(h_out * 2, 2)

        self.loss_fun = nn.CrossEntropyLoss()

    @staticmethod
    def get_sequential(ind, hidd, outd):
        return nn.Sequential(
            # nn.Dropout(0.2),
            nn.Linear(ind, hidd),
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(hidd, outd),
            nn.ReLU(),
            # nn.Dropout(0.2)
        )

    def forward(self, sent1_feat, mention1_feat, sent2_feat, mention2_feat):
        sent1_ment1 = self.get_attend_out(sent1_feat, mention1_feat)
        sent2_ment2 = self.get_attend_out(sent2_feat, mention2_feat)

        output = self.last_layer(torch.cat((sent1_ment1, sent2_ment2), dim=1))

        return output

    def get_attend_out(self, sent1_feat, mention1_feat):

        # Attend (3.1)
        attend_sent_out1 = self.F(sent1_feat)
        attend_ment_out1 = self.F(mention1_feat)

        eij1 = torch.bmm(attend_sent_out1, attend_ment_out1.transpose(1, 2))
        eij2 = eij1.transpose(1, 2)

        eij1_soft = F.softmax(eij1, dim=2)
        eij2_soft = F.softmax(eij2, dim=2)

        alpha = torch.bmm(eij2_soft, sent1_feat)
        beta = torch.bmm(eij1_soft, mention1_feat)

        # Compare (3.2)
        compare_i = torch.cat((sent1_feat, beta), dim=2)
        compare_j = torch.cat((mention1_feat, alpha), dim=2)
        v1_i = self.G(compare_i)
        v2_j = self.G(compare_j)

        # Aggregate (3.3)
        v1_sum = torch.sum(v1_i, dim=1)
        v2_sum = torch.sum(v2_j, dim=1)
        output_tolast = self.H(torch.cat((v1_sum, v2_sum), dim=1))
        return output_tolast

    def predict(self, sent1_feat, mention1_feat, sent2_feat, mention2_feat):
        output = self.__call__(sent1_feat, mention1_feat, sent2_feat, mention2_feat)
        output = F.softmax(output, dim=1)
        _, prediction = torch.max(output, 1)
        return prediction
