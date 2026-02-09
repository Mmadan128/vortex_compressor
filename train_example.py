"""
Training script for Vortex-Codec on binary data.

Trains the compressive transformer on structured binary datasets.
Supports various data formats including detector events, sensor telemetry,
network captures, and system logs.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import argparse
import time
from tqdm import tqdm

from vortex.io import ByteDataset
from vortex.core import VortexCodec
from vortex.utils.metrics import compute_bpd, cross_entropy_loss


def train_epoch(model, loader, optimizer, device, grad_clip=1.0, log_interval=100):
    """Train for one epoch with progress bar."""
    model.train()
    total_loss = 0
    total_bpd = 0
    num_batches = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        
        logits, _ = model(batch)
        
        loss = cross_entropy_loss(
            logits[:, :-1],
            batch[:, 1:]
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        
        with torch.no_grad():
            bpd = compute_bpd(logits[:, :-1], batch[:, 1:])
        
        total_loss += loss.item()
        total_bpd += bpd
        num_batches += 1
        
        if (batch_idx + 1) % log_interval == 0:
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'bpd': f'{bpd:.4f}'
            })
    
    return total_loss / num_batches, total_bpd / num_batches


def evaluate(model, loader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    total_bpd = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            
            logits, _ = model(batch)
            
            loss = cross_entropy_loss(
                logits[:, :-1],
                batch[:, 1:]
            )
            
            bpd = compute_bpd(logits[:, :-1], batch[:, 1:])
            
            total_loss += loss.item()
            total_bpd += bpd
            num_batches += 1
    
    return total_loss / num_batches, total_bpd / num_batches


def main():
    parser = argparse.ArgumentParser(description="Train Vortex-Codec on binary data")
    parser.add_argument('--data', type=str, required=True,
                       help='Path to binary training data')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (overrides config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (overrides config)')
    parser.add_argument('--output', type=str, default='checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--eval-split', type=float, default=0.1,
                       help='Fraction of data for validation')
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    device = torch.device(config['hardware']['device'] if torch.cuda.is_available() else 'cpu')
    
    if args.name is None:
        args.name = Path(args.data).stem
    
    print("=" * 70)
    print(f"Training Vortex-Codec on {args.data}")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Experiment: {args.name}")
    print(f"Output: {args.output}")
    
    print("\nLoading dataset...")
    full_dataset = ByteDataset(
        file_path=args.data,
        window_size=config['dataset']['window_size'],
        stride=config['dataset']['stride']
    )
    
    val_size = int(len(full_dataset) * args.eval_split)
    train_size = len(full_dataset) - val_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    print(f"Total samples: {len(full_dataset):,}")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"File size: {full_dataset.total_bytes:,} bytes")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print("\nInitializing model...")
    model = VortexCodec(
        **config['model'],
        **config['compressive_memory']
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Model size: ~{num_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    start_epoch = 0
    best_val_bpd = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_bpd = checkpoint.get('best_val_bpd', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    output_dir = Path(args.output) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        epoch_start = time.time()
        train_loss, train_bpd = train_epoch(
            model, train_loader, optimizer, device,
            grad_clip=config['training']['grad_clip'],
            log_interval=config['logging']['log_interval']
        )
        epoch_time = time.time() - epoch_start
        
        val_loss, val_bpd = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f} | Train BPD: {train_bpd:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val BPD:   {val_bpd:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        
        is_best = val_bpd < best_val_bpd
        if is_best:
            best_val_bpd = val_bpd
            print(f"  ✓ New best validation BPD: {best_val_bpd:.4f}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'train_bpd': train_bpd,
            'val_loss': val_loss,
            'val_bpd': val_bpd,
            'best_val_bpd': best_val_bpd,
            'config': config
        }
        
        if (epoch + 1) % config['logging']['save_interval'] == 0 or is_best:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")
        
        if is_best:
            best_path = output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            model.save(output_dir / "best_weights.pt")
    
    print("\n" + "=" * 70)
    print("Training Complete")
    print("=" * 70)
    print(f"Best validation BPD: {best_val_bpd:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
