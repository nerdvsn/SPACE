#!/usr/bin/env python3
"""
HADAR PINN v2 - Training

Physics-Informed Neural Network für spektrale Rekonstruktion.
1 Band Input → 54 Bänder Output

Verwendung:
    python train.py --config ./prepared_data/config.json --epochs 100
"""

import os
import sys
import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Füge Projekt-Root zu Path hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Importiere aus lokalen Modulen (nicht aus Unterordnern)
from model import create_pinn_model, PINN_Loss_v2, EvaluationMetrics
from dataset import HeatCubeDataset


class Trainer:
    """Trainer für HADAR PINN v2."""
    
    def __init__(self, model, train_loader, val_loader, loss_fn, 
                 optimizer, scheduler, device, output_dir, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.output_dir = output_dir
        self.config = config
        
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_recon': [], 'val_recon': [],
            'train_sam': [], 'val_sam': [],
            'train_smooth': [], 'val_smooth': [],
            'val_psnr': [], 'val_residual_mean': [],
            'lr': []
        }
    
    def train_epoch(self):
        """Trainiere eine Epoche."""
        self.model.train()
        
        total_loss = 0
        total_recon = 0
        total_sam = 0
        total_smooth = 0
        n_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Train', leave=False)
        for input_band, full_cube in pbar:
            input_band = input_band.to(self.device)
            full_cube = full_cube.to(self.device)
            
            self.optimizer.zero_grad()

            # Forward (Modell bekommt nur input_band, rekonstruiert alle 54 Bänder)
            S_pred, params = self.model(input_band)

            # Loss (full_cube ist Ground Truth)
            loss, loss_dict = self.loss_fn(S_pred, full_cube, params, compute_residual=False)
            
            # Backward
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate
            total_loss += loss.item()
            total_recon += loss_dict['reconstruction'].item()
            total_sam += loss_dict['sam'].item()
            total_smooth += loss_dict['smoothness'].item()
            n_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {
            'loss': total_loss / n_batches,
            'reconstruction': total_recon / n_batches,
            'sam': total_sam / n_batches,
            'smoothness': total_smooth / n_batches,
        }
    
    def validate(self):
        """Validiere das Modell."""
        self.model.eval()
        
        total_loss = 0
        total_recon = 0
        total_sam = 0
        total_smooth = 0
        total_psnr = 0
        total_residual = 0
        n_batches = 0
        
        with torch.no_grad():
            for input_band, full_cube in self.val_loader:
                input_band = input_band.to(self.device)
                full_cube = full_cube.to(self.device)
                
                # Forward (Modell bekommt nur input_band)
                S_pred, params = self.model(input_band)

                # Loss (full_cube ist Ground Truth)
                loss, loss_dict = self.loss_fn(S_pred, full_cube, params, compute_residual=True)
                
                # Metriken
                psnr = EvaluationMetrics.psnr(S_pred, full_cube)
                
                # Accumulate
                total_loss += loss.item()
                total_recon += loss_dict['reconstruction'].item()
                total_sam += loss_dict['sam'].item()
                total_smooth += loss_dict['smoothness'].item()
                total_psnr += psnr
                total_residual += loss_dict['residual_abs_mean']
                n_batches += 1
        
        return {
            'loss': total_loss / n_batches,
            'reconstruction': total_recon / n_batches,
            'sam': total_sam / n_batches,
            'smoothness': total_smooth / n_batches,
            'psnr': total_psnr / n_batches,
            'residual_mean': total_residual / n_batches,
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """Speichere Checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config,
        }
        
        # Letzter Checkpoint
        torch.save(checkpoint, os.path.join(self.output_dir, 'last_checkpoint.pth'))
        
        # Bester Checkpoint
        if is_best:
            torch.save(checkpoint, os.path.join(self.output_dir, 'best_model.pth'))
    
    def train(self, num_epochs, print_every=1, save_every=10):
        """Haupttrainingsschleife."""
        print(f"\n{'='*60}")
        print(f"Training für {num_epochs} Epochen")
        print(f"Device: {self.device}")
        print(f"Output: {self.output_dir}")
        print(f"{'='*60}\n")
        
        for epoch in range(num_epochs):
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # History
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_recon'].append(train_metrics['reconstruction'])
            self.history['train_sam'].append(train_metrics['sam'])
            self.history['train_smooth'].append(train_metrics['smoothness'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_recon'].append(val_metrics['reconstruction'])
            self.history['val_sam'].append(val_metrics['sam'])
            self.history['val_smooth'].append(val_metrics['smoothness'])
            self.history['val_psnr'].append(val_metrics['psnr'])
            self.history['val_residual_mean'].append(val_metrics['residual_mean'])
            self.history['lr'].append(current_lr)
            
            # Best Model?
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
            
            # Print
            if (epoch + 1) % print_every == 0 or epoch == 0:
                print(f"Epoch {epoch+1:4d}/{num_epochs} | "
                      f"Train: {train_metrics['loss']:.6f} | "
                      f"Val: {val_metrics['loss']:.6f} | "
                      f"PSNR: {val_metrics['psnr']:.2f} dB | "
                      f"SAM: {val_metrics['sam']*180/np.pi:.2f}° | "
                      f"Residual: {val_metrics['residual_mean']:.6f} | "
                      f"LR: {current_lr:.6f}" + 
                      (" *" if is_best else ""))
            
            # Save
            if (epoch + 1) % save_every == 0 or is_best:
                self.save_checkpoint(epoch + 1, is_best)
        
        # Final Save
        self.save_checkpoint(num_epochs)
        
        # History speichern
        np.savez(
            os.path.join(self.output_dir, 'history.npz'),
            **{k: np.array(v) for k, v in self.history.items()}
        )
        
        print(f"\n{'='*60}")
        print(f"Training abgeschlossen!")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}")


def main(args):
    print("=" * 60)
    print("HADAR PINN v2 - Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Lade Config
    print(f"\nLade Config: {args.config}")
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    config_dir = os.path.dirname(args.config)
    
    # Berechne Wellenzahl aus Index (720 + idx * 10)
    input_wavenumber = 720 + config['input_band_idx'] * 10
    print(f"  Bänder: {config['num_bands']}")
    print(f"  Input-Band: {config['input_band_idx']} ({input_wavenumber} cm⁻¹)")
    print(f"  Train/Val/Test: {config['num_train']}/{config['num_val']}/{config['num_test']}")
    
    # Lade Dateilisten
    with open(os.path.join(config_dir, 'train_files.json'), 'r') as f:
        train_files = json.load(f)
    with open(os.path.join(config_dir, 'val_files.json'), 'r') as f:
        val_files = json.load(f)
    
    # Datasets
    print(f"\nErstelle Datasets...")
    train_dataset = HeatCubeDataset(
        file_list=train_files,
        input_band_idx=config['input_band_idx'],
        patch_size=args.patch_size,
        patches_per_cube=args.patches_per_cube,
        augment=True,
        stats=config['statistics']
    )
    
    val_dataset = HeatCubeDataset(
        file_list=val_files,
        input_band_idx=config['input_band_idx'],
        patch_size=args.patch_size,
        patches_per_cube=args.patches_per_cube // 2,
        augment=False,
        stats=config['statistics']
    )
    
    print(f"  Train: {len(train_dataset)} patches")
    print(f"  Val: {len(val_dataset)} patches")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Modell
    print(f"\nErstelle PINN Modell...")
    wavenumbers = np.array(config['wavenumbers'])
    
    model = create_pinn_model(
        wavenumbers=wavenumbers,
        input_band_idx=config['input_band_idx'],
        dim_latent=args.dim_latent,
        hidden_dim=args.hidden_dim,
        T_min=args.T_min,
        T_max=args.T_max,
        include_atmosphere=args.include_atmosphere
    )
    model = model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameter: {n_params:,}")
    print(f"  Latent Dim: {args.dim_latent}")
    print(f"  T Range: [{args.T_min}, {args.T_max}] K")
    
    # Loss
    loss_fn = PINN_Loss_v2(
        lambda_recon=args.lambda_recon,
        lambda_sam=args.lambda_sam,
        lambda_smooth=args.lambda_smooth,
        lambda_physics=args.lambda_physics,
        lambda_temp=args.lambda_temp
    )
    
    print(f"\nLoss Weights:")
    print(f"  Reconstruction: {args.lambda_recon}")
    print(f"  SAM: {args.lambda_sam}")
    print(f"  Smoothness: {args.lambda_smooth}")
    print(f"  Physics: {args.lambda_physics}")
    print(f"  Temperature: {args.lambda_temp}")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    # Output Directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"pinn_v2_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Speichere Training-Config
    train_config = {
        'data_config': config,
        'input_band_idx': config['input_band_idx'],  # Direkt für evaluate.py
        'statistics': config.get('statistics', None),  # Direkt für evaluate.py
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'patch_size': args.patch_size,
        'lr': args.lr,
        'dim_latent': args.dim_latent,
        'hidden_dim': args.hidden_dim,
        'T_min': args.T_min,
        'T_max': args.T_max,
        'lambda_recon': args.lambda_recon,
        'lambda_sam': args.lambda_sam,
        'lambda_smooth': args.lambda_smooth,
        'lambda_physics': args.lambda_physics,
        'lambda_temp': args.lambda_temp,
    }
    
    with open(os.path.join(output_dir, 'train_config.json'), 'w') as f:
        json.dump(train_config, f, indent=2)
    
    print(f"\nOutput: {output_dir}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        config=train_config
    )
    
    # Training
    trainer.train(
        num_epochs=args.epochs,
        print_every=args.print_every,
        save_every=args.save_every
    )
    
    print(f"\nModelle gespeichert in: {output_dir}")
    print(f"\nNächster Schritt - Evaluation:")
    print(f"    python evaluate.py --model {output_dir}/best_model.pth --config {args.config}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HADAR PINN v2 Training')
    
    # Data
    parser.add_argument('--config', type=str, required=True,
                        help='Pfad zur config.json von prepare_data.py')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output-Verzeichnis')
    
    # Model
    parser.add_argument('--dim_latent', type=int, default=16,
                        help='Latente Dimension für Emissivität')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Versteckte Dimension')
    parser.add_argument('--T_min', type=float, default=260.0,
                        help='Minimale Temperatur [K]')
    parser.add_argument('--T_max', type=float, default=320.0,
                        help='Maximale Temperatur [K]')
    parser.add_argument('--include_atmosphere', action='store_true',
                        help='Atmosphäre modellieren')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Anzahl Epochen')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch Size')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch Size')
    parser.add_argument('--patches_per_cube', type=int, default=16,
                        help='Patches pro Cube')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning Rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight Decay')
    
    # Loss Weights
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_sam', type=float, default=0.1)
    parser.add_argument('--lambda_smooth', type=float, default=0.01)
    parser.add_argument('--lambda_physics', type=float, default=0.001)
    parser.add_argument('--lambda_temp', type=float, default=0.001)
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    main(args)
