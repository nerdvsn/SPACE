"""
HADAR PINN - Dataset Module

Lädt und verarbeitet HADAR Heat Cubes für das Training.

Input:  1 Band (z.B. Band 27)
Target: Alle 54 Bänder
"""

import os
import json
import numpy as np
import scipy.io as scio
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader


class HeatCubeDataset(Dataset):
    """
    Dataset für HADAR Heat Cubes.
    
    Lädt Heat Cubes und extrahiert:
    - Input: 1 Band (input_band_idx)
    - Target: Alle 54 Bänder
    """
    
    def __init__(self,
                 file_list,
                 input_band_idx=27,
                 patch_size=128,
                 patches_per_cube=32,
                 augment=False,
                 normalize=True,
                 stats=None):
        """
        Args:
            file_list: Liste von Pfaden zu .mat Dateien
            input_band_idx: Index des Input-Bands (0-53)
            patch_size: Größe der Patches
            patches_per_cube: Anzahl Patches pro Cube pro Epoche
            augment: Data Augmentation
            normalize: Daten normalisieren
            stats: Dict mit 'mean' und 'std' für Normalisierung
        """
        self.file_list = file_list
        self.input_band_idx = input_band_idx
        self.patch_size = patch_size
        self.patches_per_cube = patches_per_cube
        self.augment = augment
        self.normalize = normalize
        
        # Statistiken
        if stats is not None:
            self.mean = np.array(stats['mean'], dtype=np.float32)
            self.std = np.array(stats['std'], dtype=np.float32)
        else:
            self.mean = None
            self.std = None
        
        # Cache für geladene Cubes (optional)
        self.cube_cache = {}
        self.cache_enabled = False
        
        # Gesamtanzahl Patches
        self.total_patches = len(file_list) * patches_per_cube
    
    def enable_cache(self):
        """Aktiviere Cube-Caching (mehr RAM, schneller)."""
        self.cache_enabled = True
    
    def disable_cache(self):
        """Deaktiviere Cube-Caching."""
        self.cache_enabled = False
        self.cube_cache = {}
    
    def _load_cube(self, filepath):
        """Lade einen Heat Cube."""
        if self.cache_enabled and filepath in self.cube_cache:
            return self.cube_cache[filepath]
        
        data = scio.loadmat(filepath)
        
        # Finde den Cube (meist 'S' genannt)
        cube = None
        for key in data.keys():
            if not key.startswith('__'):
                arr = data[key]
                if isinstance(arr, np.ndarray) and len(arr.shape) == 3:
                    cube = arr.astype(np.float32)
                    break
        
        if cube is None:
            raise ValueError(f"Kein 3D Array in {filepath} gefunden")
        
        if self.cache_enabled:
            self.cube_cache[filepath] = cube
        
        return cube
    
    def __len__(self):
        return self.total_patches
    
    def __getitem__(self, idx):
        # Bestimme Cube
        cube_idx = idx // self.patches_per_cube
        filepath = self.file_list[cube_idx]
        
        # Lade Cube
        cube = self._load_cube(filepath)  # (H, W, C)
        H, W, C = cube.shape
        
        # Zufälliger Patch
        h_start = np.random.randint(0, max(1, H - self.patch_size))
        w_start = np.random.randint(0, max(1, W - self.patch_size))
        
        h_end = min(h_start + self.patch_size, H)
        w_end = min(w_start + self.patch_size, W)
        
        patch = cube[h_start:h_end, w_start:w_end, :].copy()
        
        # Padding falls nötig
        if patch.shape[0] < self.patch_size or patch.shape[1] < self.patch_size:
            padded = np.zeros((self.patch_size, self.patch_size, C), dtype=np.float32)
            padded[:patch.shape[0], :patch.shape[1], :] = patch
            patch = padded
        
        # Transpose zu (C, H, W)
        patch = patch.transpose(2, 0, 1)
        
        # Augmentation
        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=2).copy()
            # Vertical flip
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=1).copy()
            # 90° Rotation (optional)
            if np.random.rand() > 0.5:
                patch = np.rot90(patch, k=1, axes=(1, 2)).copy()
        
        # Extrahiere Input Band
        input_band = patch[self.input_band_idx:self.input_band_idx+1, :, :]  # (1, H, W)
        
        # Target ist das ganze Spektrum
        target = patch  # (54, H, W)
        
        # Normalisierung
        if self.normalize and self.mean is not None:
            input_band = (input_band - self.mean[self.input_band_idx]) / (self.std[self.input_band_idx] + 1e-8)
            # Target nicht normalisieren - wir wollen die echten Werte rekonstruieren
        
        # Zu Tensor
        input_band = torch.from_numpy(input_band)
        target = torch.from_numpy(target)
        
        return input_band, target


def compute_statistics(file_list, num_samples=10):
    """
    Berechne Statistiken (Mean, Std) über die Daten.
    
    Args:
        file_list: Liste von Cube-Pfaden
        num_samples: Wie viele Cubes für Statistik
    
    Returns:
        Dict mit 'mean' und 'std' (jeweils Array der Länge 54)
    """
    all_data = []
    
    sample_files = file_list[:min(num_samples, len(file_list))]
    
    for filepath in sample_files:
        data = scio.loadmat(filepath)
        
        for key in data.keys():
            if not key.startswith('__'):
                arr = data[key]
                if isinstance(arr, np.ndarray) and len(arr.shape) == 3:
                    # Sample random pixels
                    H, W, C = arr.shape
                    n_samples = min(5000, H * W)
                    flat = arr.reshape(-1, C)
                    indices = np.random.choice(len(flat), n_samples, replace=False)
                    all_data.append(flat[indices])
                    break
    
    all_data = np.concatenate(all_data, axis=0)
    
    mean = all_data.mean(axis=0).astype(np.float32)
    std = all_data.std(axis=0).astype(np.float32)
    
    return {'mean': mean.tolist(), 'std': std.tolist()}


def create_data_loaders(config_path, batch_size=8, num_workers=4, patch_size=128):
    """
    Erstelle DataLoaders aus Konfigurationsdatei.
    
    Args:
        config_path: Pfad zur data_config.json
        batch_size: Batch-Größe
        num_workers: Anzahl Worker
        patch_size: Patch-Größe
    
    Returns:
        train_loader, val_loader, test_loader, config
    """
    # Lade Konfiguration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    base_dir = os.path.dirname(config_path)
    
    # Lade Dateilisten
    with open(os.path.join(base_dir, 'train_files.json'), 'r') as f:
        train_files = json.load(f)
    with open(os.path.join(base_dir, 'val_files.json'), 'r') as f:
        val_files = json.load(f)
    with open(os.path.join(base_dir, 'test_files.json'), 'r') as f:
        test_files = json.load(f)
    
    # Statistiken
    stats = config.get('statistics', None)
    
    # Input Band
    input_band_idx = config.get('input_band_idx', 27)
    
    # Datasets
    train_dataset = HeatCubeDataset(
        file_list=train_files,
        input_band_idx=input_band_idx,
        patch_size=patch_size,
        patches_per_cube=32,
        augment=True,
        normalize=True,
        stats=stats
    )
    
    val_dataset = HeatCubeDataset(
        file_list=val_files,
        input_band_idx=input_band_idx,
        patch_size=patch_size,
        patches_per_cube=16,
        augment=False,
        normalize=True,
        stats=stats
    )
    
    test_dataset = HeatCubeDataset(
        file_list=test_files,
        input_band_idx=input_band_idx,
        patch_size=patch_size,
        patches_per_cube=16,
        augment=False,
        normalize=True,
        stats=stats
    )
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, config


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HADAR PINN - Dataset Test")
    print("=" * 60)
    
    # Simuliere Daten-Test ohne echte Daten
    print("\nSimulierter Test (ohne echte Daten):")
    
    # Erstelle Dummy-Statistiken
    stats = {
        'mean': [0.1] * 54,
        'std': [0.01] * 54
    }
    
    print(f"  Input Band Index: 27")
    print(f"  Patch Size: 128")
    print(f"  Mean Shape: {len(stats['mean'])}")
    print(f"  Std Shape: {len(stats['std'])}")
    
    print("\n" + "=" * 60)
    print("Dataset Test abgeschlossen!")
    print("=" * 60)
