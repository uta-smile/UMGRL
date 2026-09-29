import torch
import torch.nn.functional as F

from .base import BaseLoss, gather_and_scale_wrapper


class TripletLoss(BaseLoss):
    def __init__(self, margin, loss_term_weight=1.0):
        super(TripletLoss, self).__init__(loss_term_weight)
        self.margin = margin

    @gather_and_scale_wrapper
    def forward(self, embeddings, labels):
        embeddings = embeddings.permute(
            2, 0, 1).contiguous().float()

        ref_embed, ref_label = embeddings, labels
        dist = self.ComputeDistance(embeddings, ref_embed)
        mean_dist = dist.mean((1, 2))
        ap_dist, an_dist = self.Convert2Triplets(labels, ref_label, dist)
        dist_diff = (ap_dist - an_dist).view(dist.size(0), -1)
        loss = F.relu(dist_diff + self.margin)

        hard_loss = torch.max(loss, -1)[0]
        loss_avg, loss_num = self.AvgNonZeroReducer(loss)

        self.info.update({
            'loss': loss_avg.detach().clone(),
            'hard_loss': hard_loss.detach().clone(),
            'loss_num': loss_num.detach().clone(),
            'mean_dist': mean_dist.detach().clone()})

        return loss_avg, self.info

    def AvgNonZeroReducer(self, loss):
        eps = 1.0e-9
        loss_sum = loss.sum(-1)
        loss_num = (loss != 0).sum(-1).float()

        loss_avg = loss_sum / (loss_num + eps)
        loss_avg[loss_num == 0] = 0
        return loss_avg, loss_num

    def ComputeDistance(self, x, y):
        x2 = torch.sum(x ** 2, -1).unsqueeze(2)
        y2 = torch.sum(y ** 2, -1).unsqueeze(1)
        inner = x.matmul(y.transpose(1, 2))
        dist = x2 + y2 - 2 * inner
        dist = torch.sqrt(F.relu(dist))
        return dist

    def Convert2Triplets(self, row_labels, clo_label, dist):
        matches = (row_labels.unsqueeze(1) ==
                   clo_label.unsqueeze(0)).bool()
        diffenc = torch.logical_not(matches)
        p, n, _ = dist.size()
        ap_dist = dist[:, matches].view(p, n, -1, 1)
        an_dist = dist[:, diffenc].view(p, n, 1, -1)
        return ap_dist, an_dist
