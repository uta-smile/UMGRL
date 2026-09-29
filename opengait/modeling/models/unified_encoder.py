import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple


MODALITIES = ['seg', 'pose', 'cloud']


def resample_cloud_tokens(z_cloud: torch.Tensor, target_parts: int = 16) -> torch.Tensor:
    return F.interpolate(z_cloud, size=target_parts, mode='linear', align_corners=False)


class SharedEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 num_parts: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_parts = num_parts
        self.part_embed = nn.Parameter(torch.randn(1, num_parts, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.permute(0, 2, 1)
        x = x + self.part_embed[:, :x.size(1), :]
        x = self.transformer(x)
        x = self.out_proj(x)
        return x.permute(0, 2, 1)


class Decomposition(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model
        self.shared_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.private_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        for module in [self.shared_head, self.private_head]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = z.permute(0, 2, 1)
        c = self.shared_head(x).permute(0, 2, 1)
        r = self.private_head(x).permute(0, 2, 1)
        return c, r


class GatedRecomposition(nn.Module):
    def __init__(self, d_model: int = 256):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
            nn.Sigmoid()
        )
        for m in self.gate_mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, c: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        gamma = self.gate_mlp(c.mean(dim=-1)).unsqueeze(-1)
        return c + gamma * r


class ModalityImputer(nn.Module):
    def __init__(self, d_model: int = 256, num_parts: int = 16,
                 n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_parts = num_parts
        self.part_queries = nn.ParameterDict({
            m: nn.Parameter(torch.randn(num_parts, d_model) * 0.02) for m in MODALITIES
        })
        self.input_proj = nn.Linear(d_model, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, d_model)
        for m in [self.input_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    @staticmethod
    def prepare_visible_tokens(y: Dict[str, torch.Tensor],
                               visible_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ref = next(iter(y.values()))
        B, C, P = ref.shape
        tokens, pad = [], []
        for m_idx, modal in enumerate(MODALITIES):
            vis = visible_mask[:, m_idx]
            if modal in y:
                tok = (y[modal] * vis.view(-1, 1, 1).to(ref.dtype)).permute(0, 2, 1)
            else:
                tok = ref.new_zeros(B, P, C)
            tokens.append(tok)
            pad.append((~vis).unsqueeze(1).expand(B, P))
        return torch.cat(tokens, dim=1), torch.cat(pad, dim=1)

    def forward(self, visible_tokens: torch.Tensor, targets: List[str],
                key_padding_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        B = visible_tokens.size(0)
        memory = self.input_proj(visible_tokens)
        y_hat = {}
        for modal in targets:
            queries = self.part_queries[modal].unsqueeze(0).expand(B, -1, -1)
            decoded = self.decoder(queries, memory, memory_key_padding_mask=key_padding_mask)
            y_hat[modal] = self.out_proj(decoded).permute(0, 2, 1)
        return y_hat


def dropout_sampler(available: List[str],
                    drop_probs: Tuple[float, float, float] = (0.2, 0.7, 0.1),
                    two_modal_drop_probs: Tuple[float, float] = (0.3, 0.7)) -> List[str]:
    n = len(available)
    if n <= 1:
        return list(available)
    u = torch.rand(1).item()
    if n == 2:
        keep_all, _ = two_modal_drop_probs
        if u < keep_all:
            return list(available)
        drop = int(torch.randint(0, 2, (1,)).item())
        return [m for i, m in enumerate(available) if i != drop]
    keep_all, drop_one, _ = drop_probs
    if u < keep_all:
        return list(available)
    if u < keep_all + drop_one:
        drop = int(torch.randint(0, n, (1,)).item())
        return [m for i, m in enumerate(available) if i != drop]
    keep = int(torch.randint(0, n, (1,)).item())
    return [available[keep]]


class UnifiedEncoder(nn.Module):
    def __init__(self, d_model: int = 256, n_heads: int = 4, n_layers: int = 2,
                 num_parts: int = 16, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_parts = num_parts
        self.modality_names = MODALITIES
        self.encoder = SharedEncoder(d_model, n_heads, n_layers, num_parts, dropout)
        self.decomposition = Decomposition(d_model)
        self.recomposition = GatedRecomposition(d_model)
        self.imputer = ModalityImputer(d_model, num_parts, n_heads, n_layers=2, dropout=0.1)

    def _inputs(self, z_seg, z_pose, z_cloud) -> Dict[str, torch.Tensor]:
        inputs = {'seg': z_seg, 'pose': z_pose, 'cloud': z_cloud}
        if z_cloud is not None and z_cloud.size(-1) != self.num_parts:
            inputs['cloud'] = resample_cloud_tokens(z_cloud, self.num_parts)
        return {m: z for m, z in inputs.items() if z is not None}

    def forward(self, z_seg: Optional[torch.Tensor] = None,
                z_pose: Optional[torch.Tensor] = None,
                z_cloud: Optional[torch.Tensor] = None) -> Dict:
        inputs = self._inputs(z_seg, z_pose, z_cloud)
        if not inputs:
            raise ValueError('At least one modality must be provided')
        ref = next(iter(inputs.values()))
        B = ref.size(0)

        z_tilde, c, r, y = {}, {}, {}, {}
        for modal, z in inputs.items():
            z_tilde[modal] = self.encoder(z)
            c[modal], r[modal] = self.decomposition(z_tilde[modal])
            y[modal] = self.recomposition(c[modal], r[modal])

        available_mask = torch.zeros(B, len(MODALITIES), dtype=torch.bool, device=ref.device)
        c_list = []
        for m_idx, modal in enumerate(MODALITIES):
            if modal in c:
                available_mask[:, m_idx] = True
                c_list.append(c[modal].permute(0, 2, 1))
            else:
                c_list.append(ref.new_zeros(B, self.num_parts, self.d_model))

        return {
            'z_tilde': z_tilde,
            'c': c,
            'r': r,
            'y': y,
            'C_out': torch.stack(c_list, dim=1),
            'available_mask': available_mask,
        }

    def forward_teacher_student(self, z_seg: torch.Tensor,
                                z_pose: Optional[torch.Tensor] = None,
                                z_cloud: Optional[torch.Tensor] = None,
                                drop_probs: Tuple[float, float, float] = (0.2, 0.7, 0.1),
                                two_modal_drop_probs: Tuple[float, float] = (0.3, 0.7)) -> Dict:
        out = self.forward(z_seg, z_pose, z_cloud)
        y = out['y']
        y_target = {m: v.detach() for m, v in y.items()}

        available = [m for m in MODALITIES if m in y]
        B = next(iter(y.values())).size(0)
        device = next(iter(y.values())).device
        visible_mask = torch.zeros(B, len(MODALITIES), dtype=torch.bool, device=device)
        for b in range(B):
            for m in dropout_sampler(available, drop_probs, two_modal_drop_probs):
                visible_mask[b, MODALITIES.index(m)] = True

        missing_mask = {m: ~visible_mask[:, MODALITIES.index(m)] for m in available}
        targets = [m for m in available if missing_mask[m].any()]

        y_hat = {}
        if targets:
            tokens, pad = self.imputer.prepare_visible_tokens(y, visible_mask)
            y_hat = self.imputer(tokens, targets, key_padding_mask=pad)

        out.update({
            'y_hat': y_hat,
            'y_target': y_target,
            'visible_mask': visible_mask,
            'missing_mask': missing_mask,
            'missing_modalities_in_batch': targets,
        })
        return out

    def get_final_representation(self, z_seg: Optional[torch.Tensor] = None,
                                 z_pose: Optional[torch.Tensor] = None,
                                 z_cloud: Optional[torch.Tensor] = None,
                                 impute_missing: bool = True) -> Dict[str, torch.Tensor]:
        out = self.forward(z_seg, z_pose, z_cloud)
        y_final = dict(out['y'])
        missing = [m for m in MODALITIES if m not in y_final]
        if impute_missing and missing:
            tokens, pad = self.imputer.prepare_visible_tokens(out['y'], out['available_mask'])
            y_final.update(self.imputer(tokens, missing, key_padding_mask=pad))
        return y_final
