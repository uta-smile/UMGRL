import os
import sys
import json
import argparse
import importlib.util

import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def import_module_from_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


embedding_dataset = import_module_from_path(
    'embedding_dataset', os.path.join(PROJECT_ROOT, 'opengait/data/embedding_dataset.py'))
unified_encoder = import_module_from_path(
    'unified_encoder', os.path.join(PROJECT_ROOT, 'opengait/modeling/models/unified_encoder.py'))
geo_alignment = import_module_from_path(
    'geo_alignment', os.path.join(PROJECT_ROOT, 'opengait/modeling/losses/geo_alignment.py'))

UnifiedEncoder = unified_encoder.UnifiedEncoder
compute_total_loss = geo_alignment.compute_total_loss


def dataset_config(root):
    return [
        {
            'name': 'CCPG',
            'seg_root': f'{root}/CCPG/Baseline/GaitBase/embeddings',
            'pose_root': f'{root}/CCPG/DeepGaitV2/DeepGaitV2/embeddings',
            'cloud_root': None,
        },
        {
            'name': 'SUSTech1K',
            'seg_root': f'{root}/SUSTech1K/Baseline/GaitBase_SUSTech1K_SilsAligned/embeddings',
            'pose_root': f'{root}/SUSTech1K/DeepGaitV2/DeepGaitV2/embeddings',
            'cloud_root': f'{root}/SUSTech1K/LidarGaitPlusPlus/lidargaitv2/embeddings',
        },
    ]


def forward_batch(batch, model, args):
    z_seg = batch['seg_emb'].cuda(non_blocking=True)
    z_pose = batch['pose_emb'].cuda(non_blocking=True)
    z_cloud = batch['cloud_emb']
    if z_cloud is not None:
        z_cloud = z_cloud.cuda(non_blocking=True)
    with autocast('cuda', enabled=args.use_amp):
        out = model.forward_teacher_student(
            z_seg, z_pose, z_cloud,
            drop_probs=tuple(args.drop_probs),
            two_modal_drop_probs=tuple(args.two_modal_drop_probs))
        return compute_total_loss(out, args.lambda1, args.lambda2, args.lambda3) + (z_seg.size(0),)


def train_step(batch, model, args):
    total, n, logs = 0.0, 0, {}
    for key in ('batch_2m', 'batch_3m'):
        if batch[key] is None:
            continue
        loss, loss_dict, b = forward_batch(batch[key], model, args)
        total = total + loss * b
        n += b
        for k, v in loss_dict.items():
            logs[k] = logs.get(k, 0.0) + v * b
    return total / n, {k: v / n for k, v in logs.items()}


def save_checkpoint(path, model, optimizer, scheduler, scaler, iteration, cfg):
    torch.save({
        'iteration': iteration,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict() if scaler is not None else None,
        'cfg': cfg,
    }, path)


def main(args):
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = vars(args).copy()
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(cfg, f, indent=2)

    loader, dataset = embedding_dataset.get_mixed_dataloader(
        datasets_config=dataset_config(args.embedding_root),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    model = UnifiedEncoder(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_parts=args.num_parts,
    ).cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    scaler = GradScaler('cuda') if args.use_amp else None

    iteration = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cuda')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        if scaler is not None and ckpt.get('scaler') is not None:
            scaler.load_state_dict(ckpt['scaler'])
        iteration = int(ckpt['iteration'])

    model.train()
    while iteration < args.total_iter:
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, logs = train_step(batch, model, args)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            scheduler.step()
            iteration += 1

            if iteration % args.log_iter == 0:
                print(f"iter {iteration:6d} | " +
                      " | ".join(f"{k} {v:.4f}" for k, v in logs.items()) +
                      f" | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)
            if iteration % args.save_iter == 0 or iteration == args.total_iter:
                save_checkpoint(
                    os.path.join(args.save_dir, f'unified_encoder-{iteration:05d}.pt'),
                    model, optimizer, scheduler, scaler, iteration, cfg)
            if iteration >= args.total_iter:
                break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_root', type=str, default='output')
    parser.add_argument('--save_dir', type=str, default='output/UnifiedEncoder')
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_parts', type=int, default=16)
    parser.add_argument('--lambda1', type=float, default=1.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--lambda3', type=float, default=1.0)
    parser.add_argument('--drop_probs', type=float, nargs=3, default=[0.2, 0.7, 0.1])
    parser.add_argument('--two_modal_drop_probs', type=float, nargs=2, default=[0.3, 0.7])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--total_iter', type=int, default=50000)
    parser.add_argument('--milestones', type=int, nargs='+', default=[20000, 30000, 40000])
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--no_amp', action='store_false', dest='use_amp')
    parser.add_argument('--log_iter', type=int, default=100)
    parser.add_argument('--save_iter', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--resume', type=str, default=None)
    main(parser.parse_args())
