import torch
import torch.nn.functional as F

from torch import nn


class CorefModel(nn.Module):
    def __init__(self, f_in_dim, f_hid_dim, f_out_dim):
        super(CorefModel, self).__init__()

        self.project_embedd = None

        self.F = CorefModel.get_sequential(f_in_dim, f_hid_dim, f_out_dim)
        self.G = CorefModel.get_sequential(2 * f_in_dim, f_hid_dim, f_out_dim)
        self.H = CorefModel.get_sequential(4 * f_in_dim, f_hid_dim, f_out_dim)
        self.last_layer = nn.Linear(f_out_dim, 2)
        self.loss_fun = nn.CrossEntropyLoss()

    @staticmethod
    def get_sequential(ind, hidd, outd):
        return nn.Sequential(
            # nn.Dropout(0.4),
            nn.Linear(ind, hidd),
            nn.ReLU(),
            nn.Linear(hidd, outd),
            nn.ReLU(),
        )

    def forward(self, sent1_feat, mention1_feat, sent2_feat, mention2_feat):
        # Attend (3.1)
        attend_out1 = self.F(sent1_feat)
        attend_out2 = self.F(sent2_feat)

        eij1 = torch.bmm(attend_out1, attend_out2.transpose(1, 2))
        eij2 = eij1.transpose(1, 2)
        eij1_soft = F.softmax(eij1, dim=2)
        eij2_soft = F.softmax(eij2, dim=2)

        alpha = torch.bmm(eij2_soft, sent1_feat)
        beta = torch.bmm(eij1_soft, sent2_feat)

        # Compare (3.2)
        compare_i = torch.cat((sent1_feat, beta), dim=2)
        compare_j = torch.cat((sent2_feat, alpha), dim=2)
        v1_i = self.G(compare_i)
        v2_j = self.G(compare_j)

        # Aggregate (3.3)
        v1_sum = torch.sum(v1_i, dim=1)
        v2_sum = torch.sum(v2_j, dim=1)

        output_tolast = self.H(torch.cat((v1_sum, mention1_feat, v2_sum, mention2_feat), dim=1))

        output = self.last_layer(output_tolast)

        return output

    def predict(self, sent1_feat, mention1_feat, sent2_feat, mention2_feat):
        output = self.__call__(sent1_feat, mention1_feat, sent2_feat, mention2_feat)
        output = F.softmax(output, dim=0)
        _, prediction = torch.max(output, 1)
        return prediction
