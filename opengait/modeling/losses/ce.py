import torch
import torch.nn.functional as F

from .base import BaseLoss


class CrossEntropyLoss(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1,
                 class_weight=None, loss_term_weight=1.0,
                 log_accuracy=False):
        super(CrossEntropyLoss, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy
        weight = (torch.tensor(class_weight, dtype=torch.float32)
                  if class_weight is not None else None)
        self.register_buffer('class_weight', weight, persistent=False)

    def forward(self, logits, labels):
        n, c, p = logits.size()
        logits = logits.float()
        labels = labels.unsqueeze(1)
        if self.label_smooth:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p),
                weight=self.class_weight, label_smoothing=self.eps)
        else:
            loss = F.cross_entropy(
                logits*self.scale, labels.repeat(1, p),
                weight=self.class_weight)
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
