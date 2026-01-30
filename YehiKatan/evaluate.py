"""
HADAR PINN v2 - Evaluation Script

Evaluiert das trainierte Modell und erstellt Visualisierungen.

Verwendung:
    python evaluate.py --model ./outputs/pinn_v2_xxx/best_model.pth --test_cube ./data/Scene1_Street/HeatCubes/L_0001_heatcube.mat
"""

import os
import argparse
import json
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

from model import HADAR_PINN_v2
from physics import WAVENUMBERS_CM, NUM_BANDS


def load_model(checkpoint_path, device):
    """Lade trainiertes Modell."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    
    model = HADAR_PINN_v2(
        input_band_idx=config.get('input_band_idx', 27),
        latent_dim=16,
        encoder_channels=256,
        base_channels=64
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, config, checkpoint


def load_cube(filepath):
    """Lade Heat Cube."""
    data = scio.loadmat(filepath)
    
    for key in data.keys():
        if not key.startswith('__'):
            arr = data[key]
            if isinstance(arr, np.ndarray) and len(arr.shape) == 3:
                return arr.astype(np.float32)
    
    raise ValueError(f"Kein 3D Array gefunden in {filepath}")


def compute_metrics(S_pred, S_true):
    """Berechne Evaluierungsmetriken."""
    metrics = {}
    
    # MSE
    mse = F.mse_loss(S_pred, S_true).item()
    metrics['mse'] = mse
    
    # RMSE
    metrics['rmse'] = np.sqrt(mse)
    
    # MAE
    metrics['mae'] = F.l1_loss(S_pred, S_true).item()
    
    # Relative Error
    rel_error = torch.abs(S_pred - S_true) / (torch.abs(S_true) + 1e-8)
    metrics['rel_error'] = rel_error.mean().item()
    
    # PSNR
    max_val = S_true.max().item()
    metrics['psnr'] = 10 * np.log10(max_val**2 / (mse + 1e-10))
    
    # Spectral Angle Mapper (SAM)
    S_pred_norm = F.normalize(S_pred, dim=1)
    S_true_norm = F.normalize(S_true, dim=1)
    cos_sim = (S_pred_norm * S_true_norm).sum(dim=1)
    cos_sim = torch.clamp(cos_sim, -1, 1)
    sam = torch.acos(cos_sim) * 180 / np.pi
    metrics['sam'] = sam.mean().item()
    
    # Per-Band RMSE
    mse_per_band = ((S_pred - S_true) ** 2).mean(dim=(0, 2, 3))
    metrics['rmse_per_band'] = torch.sqrt(mse_per_band).cpu().numpy()
    
    return metrics


@torch.no_grad()
def evaluate_on_cube(model, cube, input_band_idx, device, patch_size=128, stats=None):
    """
    Evaluiere Modell auf einem Heat Cube.
    
    Returns:
        S_pred: Rekonstruiertes Spektrum
        params: Geschätzte Parameter
        metrics: Evaluierungsmetriken
    """
    H, W, C = cube.shape
    
    # Normalisierung
    if stats:
        input_mean = stats['mean'][input_band_idx]
        input_std = stats['std'][input_band_idx]
    else:
        input_mean = cube[:, :, input_band_idx].mean()
        input_std = cube[:, :, input_band_idx].std()
    
    # Vollbild-Rekonstruktion mit Sliding Window
    S_pred_full = np.zeros((C, H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    
    # Parameter-Maps
    T_map = np.zeros((H, W), dtype=np.float32)
    T_air_map = np.zeros((H, W), dtype=np.float32)
    epsilon_map = np.zeros((C, H, W), dtype=np.float32)
    tau_map = np.zeros((C, H, W), dtype=np.float32)
    
    stride = patch_size // 2
    
    for h in range(0, H - patch_size + 1, stride):
        for w in range(0, W - patch_size + 1, stride):
            # Extrahiere Patch
            patch = cube[h:h+patch_size, w:w+patch_size, :]
            patch = patch.transpose(2, 0, 1)  # (C, H, W)
            
            # Input Band
            input_band = patch[input_band_idx:input_band_idx+1, :, :]
            input_band = (input_band - input_mean) / (input_std + 1e-8)
            
            # Zu Tensor
            input_tensor = torch.from_numpy(input_band).unsqueeze(0).to(device)
            
            # Forward
            S_pred, params = model(input_tensor)
            
            # Zu NumPy
            S_pred_np = S_pred[0].cpu().numpy()
            T_np = params['T'][0, 0].cpu().numpy()
            T_air_np = params['T_air'][0, 0].cpu().numpy()
            eps_np = params['epsilon'][0].cpu().numpy()
            tau_np = params['tau'][0].cpu().numpy()
            
            # Akkumuliere
            S_pred_full[:, h:h+patch_size, w:w+patch_size] += S_pred_np
            count_map[h:h+patch_size, w:w+patch_size] += 1
            
            T_map[h:h+patch_size, w:w+patch_size] += T_np
            T_air_map[h:h+patch_size, w:w+patch_size] += T_air_np
            epsilon_map[:, h:h+patch_size, w:w+patch_size] += eps_np
            tau_map[:, h:h+patch_size, w:w+patch_size] += tau_np
    
    # Mittelung
    count_map = np.maximum(count_map, 1)
    S_pred_full /= count_map
    T_map /= count_map
    T_air_map /= count_map
    epsilon_map /= count_map
    tau_map /= count_map
    
    # Ground Truth
    S_true = cube.transpose(2, 0, 1)  # (C, H, W)
    
    # Metriken
    S_pred_tensor = torch.from_numpy(S_pred_full).unsqueeze(0)
    S_true_tensor = torch.from_numpy(S_true).unsqueeze(0)
    metrics = compute_metrics(S_pred_tensor, S_true_tensor)
    
    params_out = {
        'T': T_map,
        'T_air': T_air_map,
        'epsilon': epsilon_map,
        'tau': tau_map
    }
    
    return S_pred_full, S_true, params_out, metrics


def plot_results(S_pred, S_true, params, metrics, wavenumbers, save_path):
    """Erstelle Visualisierungen."""
    C, H, W = S_true.shape
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # --- Row 1: Spektrale Rekonstruktion verschiedener Bänder ---
    bands_to_show = [0, 17, 27, 40, 53]  # Verschiedene Bänder
    
    for i, band in enumerate(bands_to_show[:4]):
        ax = fig.add_subplot(gs[0, i])
        
        # Error Map
        error = np.abs(S_pred[band] - S_true[band])
        
        im = ax.imshow(error, cmap='hot')
        ax.set_title(f'Error Band {band}\n({wavenumbers[band]} cm⁻¹)')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    # --- Row 2: Ground Truth vs Prediction für ein Band ---
    band = 27  # Mittleres Band
    
    ax1 = fig.add_subplot(gs[1, 0])
    im1 = ax1.imshow(S_true[band], cmap='hot')
    ax1.set_title(f'Ground Truth\nBand {band}')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.046)
    
    ax2 = fig.add_subplot(gs[1, 1])
    im2 = ax2.imshow(S_pred[band], cmap='hot')
    ax2.set_title(f'Prediction\nBand {band}')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, fraction=0.046)
    
    ax3 = fig.add_subplot(gs[1, 2])
    im3 = ax3.imshow(params['T'] - 273.15, cmap='coolwarm')
    ax3.set_title('Temperature [°C]')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, fraction=0.046)
    
    ax4 = fig.add_subplot(gs[1, 3])
    im4 = ax4.imshow(params['epsilon'].mean(axis=0), cmap='viridis', vmin=0, vmax=1)
    ax4.set_title('Mean Emissivity')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, fraction=0.046)
    
    # --- Row 3: Transmittanz und Residuum ---
    ax5 = fig.add_subplot(gs[2, 0])
    im5 = ax5.imshow(params['tau'].mean(axis=0), cmap='Blues', vmin=0, vmax=1)
    ax5.set_title('Mean Transmittance τ')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046)
    
    ax6 = fig.add_subplot(gs[2, 1])
    residuum = np.abs(S_pred - S_true).mean(axis=0)
    im6 = ax6.imshow(residuum, cmap='hot')
    ax6.set_title('Mean Absolute Residuum')
    ax6.axis('off')
    plt.colorbar(im6, ax=ax6, fraction=0.046)
    
    # Spektraler Vergleich (Pixel)
    ax7 = fig.add_subplot(gs[2, 2:])
    
    # Wähle verschiedene Pixel
    pixels = [(H//4, W//4), (H//2, W//2), (3*H//4, 3*W//4)]
    colors = ['blue', 'red', 'green']
    
    for (py, px), color in zip(pixels, colors):
        ax7.plot(wavenumbers, S_true[:, py, px], '-', color=color, 
                label=f'GT ({py},{px})', linewidth=2)
        ax7.plot(wavenumbers, S_pred[:, py, px], '--', color=color,
                label=f'Pred ({py},{px})', linewidth=2, alpha=0.7)
    
    ax7.set_xlabel('Wavenumber [cm⁻¹]')
    ax7.set_ylabel('Radiance')
    ax7.set_title('Spektraler Vergleich')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # --- Row 4: Metriken und RMSE per Band ---
    ax8 = fig.add_subplot(gs[3, :2])
    ax8.plot(wavenumbers, metrics['rmse_per_band'], 'b-', linewidth=2)
    ax8.fill_between(wavenumbers, 0, metrics['rmse_per_band'], alpha=0.3)
    ax8.set_xlabel('Wavenumber [cm⁻¹]')
    ax8.set_ylabel('RMSE')
    ax8.set_title('RMSE per Spectral Band')
    ax8.grid(True, alpha=0.3)
    
    # Metriken Text
    ax9 = fig.add_subplot(gs[3, 2:])
    ax9.axis('off')
    
    metrics_text = f"""
    Evaluation Metrics
    ─────────────────────────
    MSE:        {metrics['mse']:.6f}
    RMSE:       {metrics['rmse']:.6f}
    MAE:        {metrics['mae']:.6f}
    Rel Error:  {metrics['rel_error']*100:.2f}%
    PSNR:       {metrics['psnr']:.2f} dB
    SAM:        {metrics['sam']:.2f}°
    
    ─────────────────────────
    Temperature Range:
      T:     [{params['T'].min()-273:.1f}, {params['T'].max()-273:.1f}] °C
      T_air: [{params['T_air'].min()-273:.1f}, {params['T_air'].max()-273:.1f}] °C
    
    Emissivity Range:
      ε: [{params['epsilon'].min():.3f}, {params['epsilon'].max():.3f}]
    
    Transmittance Range:
      τ: [{params['tau'].min():.3f}, {params['tau'].max():.3f}]
    """
    
    ax9.text(0.1, 0.9, metrics_text, transform=ax9.transAxes, 
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('HADAR PINN v2 - Evaluation Results', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plot gespeichert: {save_path}")


def plot_physics_analysis(S_pred, S_true, params, wavenumbers, save_path):
    """Analysiere ob Physik gelernt wurde."""
    C, H, W = S_true.shape
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Emissivitätsspektren verschiedener Pixel
    ax = axes[0, 0]
    pixels = [(H//4, W//4), (H//2, W//2), (3*H//4, 3*W//4), (H//3, 2*W//3)]
    for i, (py, px) in enumerate(pixels):
        ax.plot(wavenumbers, params['epsilon'][:, py, px], label=f'Pixel ({py},{px})')
    ax.set_xlabel('Wavenumber [cm⁻¹]')
    ax.set_ylabel('Emissivity ε')
    ax.set_title('Gelernte Emissivitätsspektren')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 2. Transmittanzspektren
    ax = axes[0, 1]
    for i, (py, px) in enumerate(pixels):
        ax.plot(wavenumbers, params['tau'][:, py, px], label=f'Pixel ({py},{px})')
    ax.set_xlabel('Wavenumber [cm⁻¹]')
    ax.set_ylabel('Transmittance τ')
    ax.set_title('Gelernte Transmittanzspektren')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 3. Temperaturverteilung
    ax = axes[0, 2]
    T_celsius = params['T'] - 273.15
    ax.hist(T_celsius.flatten(), bins=50, color='red', alpha=0.7, label='T_object')
    T_air_celsius = params['T_air'] - 273.15
    ax.hist(T_air_celsius.flatten(), bins=50, color='blue', alpha=0.7, label='T_air')
    ax.set_xlabel('Temperature [°C]')
    ax.set_ylabel('Count')
    ax.set_title('Temperaturverteilung')
    ax.legend()
    
    # 4. Residuum Analyse
    ax = axes[1, 0]
    residuum = (S_pred - S_true).flatten()
    ax.hist(residuum, bins=100, color='purple', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuum (Pred - True)')
    ax.set_ylabel('Count')
    ax.set_title(f'Residuum-Verteilung\nMean: {residuum.mean():.2e}, Std: {residuum.std():.2e}')
    
    # 5. Scatter: Predicted vs True
    ax = axes[1, 1]
    # Sample für Performance
    n_samples = min(10000, S_pred.size)
    indices = np.random.choice(S_pred.size, n_samples, replace=False)
    ax.scatter(S_true.flatten()[indices], S_pred.flatten()[indices], 
              alpha=0.1, s=1, c='blue')
    
    # Ideale Linie
    lims = [min(S_true.min(), S_pred.min()), max(S_true.max(), S_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Ideal')
    ax.set_xlabel('Ground Truth')
    ax.set_ylabel('Predicted')
    ax.set_title('Predicted vs Ground Truth')
    ax.legend()
    
    # 6. Emissivity Smoothness Check
    ax = axes[1, 2]
    # Berechne Glattheit (erste Ableitung)
    eps_diff = np.abs(np.diff(params['epsilon'], axis=0))
    mean_roughness = eps_diff.mean(axis=(1, 2))
    ax.bar(wavenumbers[:-1], mean_roughness)
    ax.set_xlabel('Wavenumber [cm⁻¹]')
    ax.set_ylabel('Mean |dε/dν|')
    ax.set_title('Emissivity Smoothness\n(niedriger = glatter)')
    
    plt.suptitle('Physics Analysis - Wurde Physik gelernt?', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Physics Analysis gespeichert: {save_path}")


def main(args):
    print("=" * 60)
    print("HADAR PINN v2 - Evaluation")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Lade Modell
    print(f"\nLade Modell: {args.model}")
    model, config, checkpoint = load_model(args.model, device)
    
    input_band_idx = config.get('input_band_idx', 27)
    print(f"  Input Band: {input_band_idx} ({WAVENUMBERS_CM[input_band_idx]} cm⁻¹)")
    print(f"  Best Val Loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    # Lade Cube
    print(f"\nLade Test Cube: {args.test_cube}")
    cube = load_cube(args.test_cube)
    print(f"  Shape: {cube.shape}")
    
    # Statistiken
    stats = config.get('statistics', None)
    
    # Evaluiere
    print(f"\nEvaluiere...")
    S_pred, S_true, params, metrics = evaluate_on_cube(
        model, cube, input_band_idx, device,
        patch_size=args.patch_size,
        stats=stats
    )
    
    # Print Metriken
    print(f"\n{'='*40}")
    print("Evaluation Metrics:")
    print(f"{'='*40}")
    print(f"  MSE:        {metrics['mse']:.6f}")
    print(f"  RMSE:       {metrics['rmse']:.6f}")
    print(f"  MAE:        {metrics['mae']:.6f}")
    print(f"  Rel Error:  {metrics['rel_error']*100:.2f}%")
    print(f"  PSNR:       {metrics['psnr']:.2f} dB")
    print(f"  SAM:        {metrics['sam']:.2f}°")
    
    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plots
    plot_results(
        S_pred, S_true, params, metrics,
        WAVENUMBERS_CM,
        os.path.join(args.output_dir, 'evaluation_results.png')
    )
    
    plot_physics_analysis(
        S_pred, S_true, params,
        WAVENUMBERS_CM,
        os.path.join(args.output_dir, 'physics_analysis.png')
    )
    
    # Speichere Metriken
    metrics_save = {k: v.tolist() if isinstance(v, np.ndarray) else v 
                   for k, v in metrics.items()}
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics_save, f, indent=2)
    
    # Speichere Parameter
    np.savez(
        os.path.join(args.output_dir, 'parameters.npz'),
        T=params['T'],
        T_air=params['T_air'],
        epsilon=params['epsilon'],
        tau=params['tau'],
        S_pred=S_pred,
        S_true=S_true
    )
    
    print(f"\nErgebnisse gespeichert in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HADAR PINN v2 Evaluation')
    
    parser.add_argument('--model', type=str, required=True,
                        help='Pfad zum trainierten Modell (.pth)')
    parser.add_argument('--test_cube', type=str, required=True,
                        help='Pfad zum Test Heat Cube')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                        help='Output-Verzeichnis')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch Size für Evaluation')
    
    args = parser.parse_args()
    main(args)
