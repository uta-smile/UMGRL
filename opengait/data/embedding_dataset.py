import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader


class EmbeddingDataset(Dataset):
    def __init__(self,
                 seg_root,
                 pose_root,
                 cloud_root=None,
                 person_ids=None,
                 ):
        self.seg_root = seg_root
        self.pose_root = pose_root
        self.cloud_root = cloud_root
        self.has_cloud = cloud_root is not None

        self.samples = []
        self._collect_samples(person_ids)

    def _collect_samples(self, person_ids):
        seg_pids = set(os.listdir(self.seg_root))
        pose_pids = set(os.listdir(self.pose_root))

        if self.has_cloud:
            cloud_pids = set(os.listdir(self.cloud_root))
            valid_pids = seg_pids & pose_pids & cloud_pids
        else:
            valid_pids = seg_pids & pose_pids

        valid_pids = {p for p in valid_pids if not p.startswith('.')}

        if person_ids is not None:
            valid_pids = valid_pids & set(person_ids)

        valid_pids = sorted(valid_pids)

        self.pid_to_idx = {pid: idx for idx, pid in enumerate(valid_pids)}
        self.idx_to_pid = {idx: pid for pid, idx in self.pid_to_idx.items()}
        self.num_persons = len(valid_pids)

        for pid in valid_pids:
            seg_pid_dir = os.path.join(self.seg_root, pid)

            if not os.path.isdir(seg_pid_dir):
                continue

            for cond in os.listdir(seg_pid_dir):
                seg_cond_dir = os.path.join(seg_pid_dir, cond)
                if not os.path.isdir(seg_cond_dir):
                    continue

                for view in os.listdir(seg_cond_dir):
                    seg_view_dir = os.path.join(seg_cond_dir, view)
                    if not os.path.isdir(seg_view_dir):
                        continue

                    seg_files = [f for f in os.listdir(seg_view_dir) if f.endswith('-emb.npy')]
                    if not seg_files:
                        continue

                    seg_path = os.path.join(seg_view_dir, seg_files[0])

                    pose_view_dir = os.path.join(self.pose_root, pid, cond, view)
                    if not os.path.exists(pose_view_dir):
                        continue

                    pose_files = [f for f in os.listdir(pose_view_dir) if f.endswith('-emb.npy')]
                    if not pose_files:
                        continue
                    pose_path = os.path.join(pose_view_dir, pose_files[0])

                    cloud_path = None
                    if self.has_cloud:
                        cloud_view_dir = os.path.join(self.cloud_root, pid, cond, view)
                        if not os.path.exists(cloud_view_dir):
                            continue

                        cloud_files = [f for f in os.listdir(cloud_view_dir) if f.endswith('-emb.npy')]
                        if not cloud_files:
                            continue
                        cloud_path = os.path.join(cloud_view_dir, cloud_files[0])

                    self.samples.append({
                        'pid': pid,
                        'pid_idx': self.pid_to_idx[pid],
                        'cond': cond,
                        'view': view,
                        'seg_path': seg_path,
                        'pose_path': pose_path,
                        'cloud_path': cloud_path,
                    })

        print(f"[EmbeddingDataset] Collected {len(self.samples)} samples from {len(valid_pids)} persons")
        print(f"[EmbeddingDataset] Has cloud: {self.has_cloud}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        seg_emb = np.load(sample['seg_path']).squeeze(0).astype(np.float32)
        pose_emb = np.load(sample['pose_path']).squeeze(0).astype(np.float32)

        seg_emb = torch.from_numpy(seg_emb)
        pose_emb = torch.from_numpy(pose_emb)

        if sample['cloud_path'] is not None:
            cloud_emb = np.load(sample['cloud_path']).squeeze(0).astype(np.float32)
            cloud_emb = torch.from_numpy(cloud_emb)
        else:
            cloud_emb = None

        return {
            'seg_emb': seg_emb,
            'pose_emb': pose_emb,
            'cloud_emb': cloud_emb,
            'pid': sample['pid_idx'],
            'pid_str': sample['pid'],
        }


class MixedEmbeddingDataset(Dataset):
    def __init__(self, datasets_config):
        self.samples = []
        self.pid_to_global_idx = {}
        global_pid_offset = 0

        for cfg in datasets_config:
            dataset = EmbeddingDataset(
                seg_root=cfg['seg_root'],
                pose_root=cfg['pose_root'],
                cloud_root=cfg.get('cloud_root'),
            )

            for sample in dataset.samples:
                local_pid = sample['pid']
                global_key = f"{cfg['name']}_{local_pid}"

                if global_key not in self.pid_to_global_idx:
                    self.pid_to_global_idx[global_key] = global_pid_offset
                    global_pid_offset += 1

                sample['pid_idx'] = self.pid_to_global_idx[global_key]
                sample['dataset'] = cfg['name']
                sample['has_cloud'] = cfg.get('cloud_root') is not None
                self.samples.append(sample)

        self.num_persons = global_pid_offset
        print(f"[MixedEmbeddingDataset] Total samples: {len(self.samples)}")
        print(f"[MixedEmbeddingDataset] Total persons: {self.num_persons}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        seg_emb = np.load(sample['seg_path']).squeeze(0).astype(np.float32)
        pose_emb = np.load(sample['pose_path']).squeeze(0).astype(np.float32)

        seg_emb = torch.from_numpy(seg_emb)
        pose_emb = torch.from_numpy(pose_emb)

        if sample['cloud_path'] is not None:
            cloud_emb = np.load(sample['cloud_path']).squeeze(0).astype(np.float32)
            cloud_emb = torch.from_numpy(cloud_emb)
        else:
            cloud_emb = None

        return {
            'seg_emb': seg_emb,
            'pose_emb': pose_emb,
            'cloud_emb': cloud_emb,
            'pid': sample['pid_idx'],
            'has_cloud': sample['has_cloud'],
            'dataset': sample['dataset'],
        }


def mixed_collate_fn(batch):
    samples_2m = [item for item in batch if not item['has_cloud']]
    samples_3m = [item for item in batch if item['has_cloud']]

    result = {
        'batch_2m': None,
        'batch_3m': None,
    }

    if samples_2m:
        result['batch_2m'] = {
            'seg_emb': torch.stack([item['seg_emb'] for item in samples_2m]),
            'pose_emb': torch.stack([item['pose_emb'] for item in samples_2m]),
            'cloud_emb': None,
            'pid': torch.tensor([item['pid'] for item in samples_2m], dtype=torch.long),
        }

    if samples_3m:
        result['batch_3m'] = {
            'seg_emb': torch.stack([item['seg_emb'] for item in samples_3m]),
            'pose_emb': torch.stack([item['pose_emb'] for item in samples_3m]),
            'cloud_emb': torch.stack([item['cloud_emb'] for item in samples_3m]),
            'pid': torch.tensor([item['pid'] for item in samples_3m], dtype=torch.long),
        }

    return result


def get_mixed_dataloader(datasets_config, batch_size=32, shuffle=True,
                         num_workers=4, drop_last=True):
    dataset = MixedEmbeddingDataset(datasets_config)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=mixed_collate_fn,
        drop_last=drop_last,
        pin_memory=True,
    )

    return loader, dataset
