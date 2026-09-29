import torch
import torch.nn.functional as F
from typing import Dict, Tuple

MODALITIES = ['seg', 'pose', 'cloud']


def compute_L_geo(C_out: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
    c_bar = C_out.mean(dim=2)
    losses = []
    for b in range(c_bar.size(0)):
        idx = available_mask[b].nonzero(as_tuple=True)[0]
        if idx.numel() < 2:
            continue
        Z_align = c_bar[b, idx].T.float()
        S = torch.linalg.svdvals(Z_align)
        q = S[0] / (S.sum() + 1e-8)
        losses.append(-torch.log(q + 1e-8))
    if losses:
        return torch.stack(losses).mean()
    return C_out.new_zeros(())


def compute_L_dir(C_out: torch.Tensor, available_mask: torch.Tensor) -> torch.Tensor:
    c_bar = C_out.mean(dim=2)
    losses = []
    for b in range(c_bar.size(0)):
        idx = available_mask[b].nonzero(as_tuple=True)[0].tolist()
        pairs = [(i, j) for k, i in enumerate(idx) for j in idx[k + 1:]]
        if not pairs:
            continue
        cos = torch.stack([F.cosine_similarity(c_bar[b, i], c_bar[b, j], dim=0) for i, j in pairs])
        losses.append(1.0 - cos.mean())
    if losses:
        return torch.stack(losses).mean()
    return C_out.new_zeros(())


def compute_L_ortho(c: Dict[str, torch.Tensor], r: Dict[str, torch.Tensor]) -> torch.Tensor:
    per_modal = []
    for modal in c:
        c_n = F.normalize(c[modal], dim=1)
        r_n = F.normalize(r[modal], dim=1)
        cross = torch.bmm(c_n.transpose(1, 2), r_n)
        per_modal.append(cross.pow(2).sum(dim=(1, 2)))
    return torch.stack(per_modal, dim=1).mean(dim=1).mean()


def compute_L_imp(y_hat: Dict[str, torch.Tensor],
                  y_target: Dict[str, torch.Tensor],
                  missing_mask: Dict[str, torch.Tensor]) -> torch.Tensor:
    ref = next(iter(y_target.values()))
    per_sample = ref.new_zeros(ref.size(0))
    for modal, pred in y_hat.items():
        err = (pred - y_target[modal]).pow(2).sum(dim=(1, 2))
        per_sample = per_sample + err * missing_mask[modal].to(err.dtype)
    return per_sample.mean()


def compute_total_loss(out: Dict,
                       lambda1: float = 1.0,
                       lambda2: float = 1.0,
                       lambda3: float = 1.0) -> Tuple[torch.Tensor, Dict[str, float]]:
    C_out = out['C_out']
    available_mask = out['available_mask']
    L_geo = compute_L_geo(C_out, available_mask)
    L_dir = compute_L_dir(C_out, available_mask)
    L_ortho = compute_L_ortho(out['c'], out['r'])
    L_repr = L_geo + lambda1 * L_dir + lambda2 * L_ortho
    if out['y_hat']:
        L_imp = compute_L_imp(out['y_hat'], out['y_target'], out['missing_mask'])
    else:
        L_imp = C_out.new_zeros(())
    L_total = L_repr + lambda3 * L_imp
    loss_dict = {
        'L_total': L_total.item(),
        'L_geo': L_geo.item(),
        'L_dir': L_dir.item(),
        'L_ortho': L_ortho.item(),
        'L_imp': L_imp.item(),
    }
    return L_total, loss_dict
