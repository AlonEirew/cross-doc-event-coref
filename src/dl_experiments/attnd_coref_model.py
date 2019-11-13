from torch import nn


class AttendCorefModel(nn.Module):
    def __init__(self, q_k_v_in_dim, q_k_v_out_dim):
        super(AttendCorefModel, self).__init__()

        self.project_embedd = None

        self.WQ1 = nn.Linear(q_k_v_in_dim, q_k_v_out_dim)
        self.WK1 = nn.Linear(q_k_v_in_dim, q_k_v_out_dim)
        self.WV1 = nn.Linear(q_k_v_in_dim, q_k_v_out_dim)

        self.loss_fun = nn.CrossEntropyLoss()

    def forward(self, sent1_feat, mention1_feat, sent2_feat, mention2_feat):
        sent1_q = self.WQ1(sent1_feat)
        sent1_k = self.WK1(sent1_feat)
        sent1_v = self.WV1(sent1_feat)

        ment1_q = self.WQ1(mention1_feat)
        ment1_k = self.WK1(mention1_feat)
        ment1_v = self.WV1(mention1_feat)

        sent2_q = self.WQ1(sent2_feat)
        sent2_k = self.WK1(sent2_feat)
        sent2_v = self.WV1(sent2_feat)

        ment2_q = self.WQ1(mention2_feat)
        ment2_k = self.WK1(mention2_feat)
        ment2_v = self.WV1(mention2_feat)

    def predict(self, sent1_feat, mention1_feat, sent2_feat, mention2_feat):
        pass
