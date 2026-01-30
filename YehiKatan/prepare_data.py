"""
HADAR PINN v2 - Data Preparation

Bereitet die HADAR-Daten für das Training vor:
1. Findet alle Heat Cubes
2. Teilt in Train/Val/Test
3. Berechnet Statistiken
4. Speichert Konfiguration

Verwendung:
    python prepare_data.py --data_root ./data --output_dir ./prepared_data
"""

import os
import argparse
import json
import numpy as np
import scipy.io as scio
from pathlib import Path
from tqdm import tqdm


def find_heatcubes(data_root):
    """Finde alle Heat Cube Dateien."""
    heatcube_files = []
    
    for mat_file in Path(data_root).rglob("*heatcube*.mat"):
        heatcube_files.append(str(mat_file))
    
    heatcube_files.sort()
    return heatcube_files


def get_scene_name(filepath):
    """Extrahiere Szenen-Name aus Pfad."""
    parts = Path(filepath).parts
    for part in parts:
        if 'Scene' in part or 'scene' in part:
            return part
    return 'Unknown'


def analyze_cube(filepath):
    """Analysiere einen Heat Cube."""
    try:
        data = scio.loadmat(filepath)
        
        for key in data.keys():
            if not key.startswith('__'):
                arr = data[key]
                if isinstance(arr, np.ndarray) and len(arr.shape) == 3:
                    return {
                        'filepath': filepath,
                        'shape': arr.shape,
                        'min': float(arr.min()),
                        'max': float(arr.max()),
                        'mean': float(arr.mean()),
                        'dtype': str(arr.dtype)
                    }
        return None
    except Exception as e:
        print(f"Fehler bei {filepath}: {e}")
        return None


def split_by_scene(files, train_ratio=0.7, val_ratio=0.15, seed=42):
    """Teile Dateien nach Szenen."""
    np.random.seed(seed)
    
    # Gruppiere nach Szene
    scenes = {}
    for f in files:
        scene = get_scene_name(f)
        if scene not in scenes:
            scenes[scene] = []
        scenes[scene].append(f)
    
    scene_names = list(scenes.keys())
    np.random.shuffle(scene_names)
    
    n = len(scene_names)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    
    train_scenes = scene_names[:n_train]
    val_scenes = scene_names[n_train:n_train + n_val]
    test_scenes = scene_names[n_train + n_val:]
    
    if len(test_scenes) == 0:
        test_scenes = val_scenes[-1:]
        val_scenes = val_scenes[:-1] if len(val_scenes) > 1 else val_scenes
    
    train_files = [f for s in train_scenes for f in scenes[s]]
    val_files = [f for s in val_scenes for f in scenes[s]]
    test_files = [f for s in test_scenes for f in scenes[s]]
    
    return train_files, val_files, test_files, {
        'train_scenes': train_scenes,
        'val_scenes': val_scenes,
        'test_scenes': test_scenes
    }


def compute_statistics(file_list, num_samples=10):
    """Berechne Mean und Std über Trainingsdaten."""
    print("Berechne Statistiken...")
    
    all_data = []
    sample_files = file_list[:min(num_samples, len(file_list))]
    
    for filepath in tqdm(sample_files, desc="Sampling"):
        try:
            data = scio.loadmat(filepath)
            
            for key in data.keys():
                if not key.startswith('__'):
                    arr = data[key]
                    if isinstance(arr, np.ndarray) and len(arr.shape) == 3:
                        H, W, C = arr.shape
                        n_samples = min(10000, H * W)
                        flat = arr.reshape(-1, C).astype(np.float32)
                        indices = np.random.choice(len(flat), n_samples, replace=False)
                        all_data.append(flat[indices])
                        break
        except Exception as e:
            print(f"  Fehler bei {filepath}: {e}")
    
    if len(all_data) == 0:
        print("WARNUNG: Keine Daten für Statistiken!")
        return {'mean': [0.1] * 54, 'std': [0.01] * 54}
    
    all_data = np.concatenate(all_data, axis=0)
    
    mean = all_data.mean(axis=0)
    std = all_data.std(axis=0)
    
    return {
        'mean': mean.tolist(),
        'std': std.tolist(),
        'global_min': float(all_data.min()),
        'global_max': float(all_data.max())
    }


def main(args):
    print("=" * 60)
    print("HADAR PINN v2 - Data Preparation")
    print("=" * 60)
    
    # 1. Finde Heat Cubes
    print(f"\n1. Suche Heat Cubes in: {args.data_root}")
    files = find_heatcubes(args.data_root)
    print(f"   Gefunden: {len(files)} Dateien")
    
    if len(files) == 0:
        print("FEHLER: Keine Heat Cubes gefunden!")
        return
    
    # Zeige Szenen
    scenes = set(get_scene_name(f) for f in files)
    print(f"   Szenen: {len(scenes)}")
    for scene in sorted(scenes):
        count = sum(1 for f in files if get_scene_name(f) == scene)
        print(f"      - {scene}: {count} Cubes")
    
    # 2. Analysiere ersten Cube
    print(f"\n2. Analysiere Datenformat...")
    info = analyze_cube(files[0])
    if info:
        print(f"   Shape: {info['shape']}")
        print(f"   Range: [{info['min']:.4f}, {info['max']:.4f}]")
        print(f"   Mean: {info['mean']:.4f}")
    
    # 3. Split
    print(f"\n3. Teile Daten (Train/Val/Test)...")
    train_files, val_files, test_files, split_info = split_by_scene(
        files, 
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    print(f"   Train: {len(train_files)} Cubes ({split_info['train_scenes']})")
    print(f"   Val:   {len(val_files)} Cubes ({split_info['val_scenes']})")
    print(f"   Test:  {len(test_files)} Cubes ({split_info['test_scenes']})")
    
    # 4. Statistiken
    print(f"\n4. Berechne Normalisierungsstatistiken...")
    stats = compute_statistics(train_files, num_samples=10)
    print(f"   Mean Range: [{min(stats['mean']):.4f}, {max(stats['mean']):.4f}]")
    print(f"   Std Range: [{min(stats['std']):.4f}, {max(stats['std']):.4f}]")
    
    # 5. Speichern
    print(f"\n5. Speichere Konfiguration...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Dateilisten
    with open(os.path.join(args.output_dir, 'train_files.json'), 'w') as f:
        json.dump(train_files, f, indent=2)
    with open(os.path.join(args.output_dir, 'val_files.json'), 'w') as f:
        json.dump(val_files, f, indent=2)
    with open(os.path.join(args.output_dir, 'test_files.json'), 'w') as f:
        json.dump(test_files, f, indent=2)
    
    # Konfiguration
    input_wavenumber = 720 + args.input_band_idx * 10  # Wellenzahl in cm⁻¹
    config = {
        'data_root': os.path.abspath(args.data_root),
        'input_band_idx': args.input_band_idx,
        'input_wavenumber': input_wavenumber,
        'num_bands': 54,
        'wavenumbers': list(range(720, 1260, 10)),
        'train_ratio': args.train_ratio,
        'val_ratio': args.val_ratio,
        'seed': args.seed,
        'num_train': len(train_files),
        'num_val': len(val_files),
        'num_test': len(test_files),
        'statistics': stats,
        'split_info': split_info,
        'cube_shape': info['shape'] if info else None
    }

    # Speichere als config.json (für train.py Kompatibilität)
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"   Gespeichert in: {args.output_dir}")
    
    # Zusammenfassung
    print("\n" + "=" * 60)
    print("Fertig! Nächster Schritt:")
    print("=" * 60)
    print(f"""
    python train.py --config {os.path.join(args.output_dir, 'config.json')} --epochs 100
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HADAR PINN v2 Data Preparation')
    
    parser.add_argument('--data_root', type=str, required=True,
                        help='Pfad zum data/ Ordner')
    parser.add_argument('--output_dir', type=str, default='./prepared_data',
                        help='Output-Verzeichnis')
    parser.add_argument('--input_band_idx', type=int, default=27,
                        help='Index des Input-Bands (0-53, default: 27 = 990 cm⁻¹)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Anteil Training')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='Anteil Validation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random Seed')
    
    args = parser.parse_args()
    main(args)
