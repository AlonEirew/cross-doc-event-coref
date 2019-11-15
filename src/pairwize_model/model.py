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
