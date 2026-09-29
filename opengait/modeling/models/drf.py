import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..base_model import BaseModel
from ..modules import (HorizontalPoolingPyramid, PackSequenceWrapper,
                       SeparateBNNecks, SeparateFCs, SetBlockWrapper)


CLASS_TO_INDEX = {'negative': 0, 'neutral': 1, 'positive': 2}


class PAVGuidedAttention(nn.Module):
    def __init__(self, channels=256, pair_num=8, metric_num=3):
        super().__init__()
        self.pair_num = pair_num
        self.metric_num = metric_num
        self.channel_attention = nn.Linear(pair_num * metric_num, channels)
        self.spatial_attention = nn.Conv1d(metric_num, 1, kernel_size=1)

    def forward(self, features, pav):
        if pav.ndim != 3 or pav.shape[1:] != (self.pair_num, self.metric_num):
            raise ValueError(
                'Expected PAV [N, {}, {}], got {}.'.format(
                    self.pair_num, self.metric_num, tuple(pav.shape)))
        if features.ndim != 3:
            raise ValueError('Expected encoded features [N, C, H], got {}.'
                             .format(tuple(features.shape)))

        channel_gate = torch.sigmoid(
            self.channel_attention(pav.flatten(1))).unsqueeze(-1)

        spatial_logits = self.spatial_attention(pav.transpose(1, 2))
        spatial_logits = F.interpolate(
            spatial_logits, size=features.size(-1), mode='linear',
            align_corners=False)
        spatial_gate = torch.sigmoid(spatial_logits)
        return features * channel_gate * spatial_gate, channel_gate, spatial_gate


class DRF(BaseModel):
    def build_network(self, model_cfg):
        self.use_pga = model_cfg.get('use_pga', True)
        self.triplet_label = model_cfg.get('triplet_label', 'identity')
        if self.triplet_label not in ('identity', 'class'):
            raise ValueError(
                "triplet_label must be 'identity' or 'class', got {}."
                .format(self.triplet_label))
        self.Backbone = SetBlockWrapper(
            self.get_backbone(model_cfg['backbone_cfg']))
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        if self.use_pga:
            self.PGA = PAVGuidedAttention(
                channels=model_cfg['SeparateFCs']['out_channels'],
                pair_num=model_cfg.get('pav_pair_num', 8),
                metric_num=model_cfg.get('pav_metric_num', 3))
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)

    @staticmethod
    def _sequence_pav(pav, seqL):
        if pav.ndim != 4:
            raise ValueError('Expected batched PAV [N,T,8,3], got {}.'
                             .format(tuple(pav.shape)))
        if seqL is None:
            return pav.mean(dim=1)

        lengths = seqL[0].detach().cpu().tolist()
        if pav.size(0) != 1:
            raise ValueError('Packed PAV must have leading dimension 1, got {}.'
                             .format(pav.size(0)))
        chunks = torch.split(pav[0], lengths, dim=0)
        return torch.stack([chunk.mean(dim=0) for chunk in chunks], dim=0)

    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs
        if len(ipts) != 2:
            raise ValueError('DRF requires [skeleton_map, PAV], got {} inputs.'
                             .format(len(ipts)))

        maps, pav_frames = ipts
        if maps.ndim == 4:
            maps = maps.unsqueeze(2)
        if maps.ndim != 5:
            raise ValueError('Expected skeleton maps [N,T,2,H,W], got {}.'
                             .format(tuple(maps.shape)))
        maps = maps.transpose(1, 2).contiguous()
        if maps.size(1) != 2:
            raise ValueError('DRF expects two skeleton-map channels, got {}.'
                             .format(maps.size(1)))

        pav = self._sequence_pav(pav_frames, seqL)
        class_ids = torch.as_tensor(
            [CLASS_TO_INDEX[label] for label in labels],
            device=maps.device, dtype=torch.long)

        encoded = self.Backbone(maps)
        encoded = self.TP(encoded, seqL, options={'dim': 2})[0]
        encoded = self.HPP(encoded)
        encoded = self.FCs(encoded)
        if self.use_pga:
            refined, channel_gate, spatial_gate = self.PGA(encoded, pav)
        else:
            refined = encoded
            channel_gate = encoded.new_ones(encoded.size(0), encoded.size(1), 1)
            spatial_gate = encoded.new_ones(encoded.size(0), 1, encoded.size(2))
        embed, logits = self.BNNecks(refined)

        return {
            'training_feat': {
                'triplet': {'embeddings': refined, 'labels': class_ids if self.triplet_label == 'class' else pids},
                'softmax': {'logits': logits, 'labels': class_ids},
            },
            'visual_summary': {
                'image/skeleton_map': rearrange(
                    maps, 'n c s h w -> (n s) c h w'),
                'scalar/attention_channel_mean': channel_gate.mean(),
                'scalar/attention_spatial_mean': spatial_gate.mean(),
            },
            'inference_feat': {
                'embeddings': logits,
            }
        }
