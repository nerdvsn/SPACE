"""
HADAR PINN Evaluation Script

Evaluiert trainierte PINN Modelle und vergleicht 1-Band vs. 5-Band Rekonstruktion.

Verwendung:
    python evaluate_pinn.py --model_1band ./outputs/pinn_1band_20260123_082956/best_model.pth \
                            --model_5band ./outputs/pinn_5band/best_model.pth \
                            --test_cube ./data/L_0001_heatcube.mat
"""

import os
import argparse
import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import torch
import torch.nn.functional as F

# Füge safe_globals für PyTorch 2.6+ hinzu
import numpy as np
torch.serialization.add_safe_globals([np._core.multiarray.scalar])

try:
    from pinn_spectral_reconstruction import HADAR_PINN, create_pinn_model
    has_hadar_pinn = True
except ImportError:
    try:
        from pinn_spectral_reconstruction import PINNSpectralReconstruction
        has_hadar_pinn = False
        print("Warnung: HADAR_PINN nicht gefunden, verwende PINNSpectralReconstruction")
    except ImportError:
        raise ImportError("Keine gültige Modellklasse in pinn_spectral_reconstruction gefunden")


def load_model(checkpoint_path, wavenumbers, emissivity_lib, device):
    """Lade ein trainiertes Modell."""
    try:
        # Versuche zuerst mit weights_only=False (PyTorch 2.6+)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        print(f"Warnung: Konnte nicht mit weights_only=False laden: {e}")
        # Fallback zur alten Methode
        checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Debug-Ausgabe um Checkpoint-Struktur zu sehen
    print(f"Checkpoint Keys: {list(checkpoint.keys())}")
    
    # Extrahiere Input-Band-Indizes
    if 'input_band_indices' in checkpoint:
        input_indices = checkpoint['input_band_indices']
    elif 'input_indices' in checkpoint:
        input_indices = checkpoint['input_indices']
    elif 'indices' in checkpoint:
        input_indices = checkpoint['indices']
    else:
        # Standardindizes basierend auf 1-Band oder 5-Band
        if '1band' in checkpoint_path.lower():
            input_indices = [27]  # Standard für 1-Band aus Ihrer Ausgabe
        elif '5band' in checkpoint_path.lower():
            # Typische 5-Band Indizes - anpassen falls nötig
            input_indices = [10, 20, 30, 40, 50]
        else:
            # Annahme: Es ist ein 1-Band Modell
            input_indices = [27]
    
    num_input_bands = len(input_indices)
    
    # Extrahiere Modellparameter
    if 'args' in checkpoint:
        hidden_dim = checkpoint['args'].get('hidden_dim', 64)
    elif 'config' in checkpoint:
        hidden_dim = checkpoint['config'].get('hidden_dim', 64)
    elif 'hidden_dim' in checkpoint:
        hidden_dim = checkpoint['hidden_dim']
    else:
        hidden_dim = 64  # Standardwert
    
    # Erstelle Modell
    if has_hadar_pinn:
        model = create_pinn_model(
            wavenumbers=wavenumbers,
            emissivity_library=emissivity_lib,
            num_input_bands=num_input_bands,
            hidden_dim=hidden_dim
        )
    else:
        # Verwende PINNSpectralReconstruction direkt
        output_dim = len(wavenumbers)
        num_materials = emissivity_lib.shape[0]
        
        model = PINNSpectralReconstruction(
            input_dim=num_input_bands,
            output_dim=output_dim,
            num_materials=num_materials,
            matlib=emissivity_lib,
            hidden_dim=hidden_dim,
            num_layers=8  # Standardwert
        )
    
    # Lade Modellgewichte
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        # Annahme: Der Checkpoint enthält direkt die Gewichte
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    print(f"Modell geladen: {checkpoint_path}")
    print(f"  Input-Bänder: {input_indices}")
    print(f"  Num Input Bands: {num_input_bands}")
    print(f"  Hidden Dim: {hidden_dim}")
    
    return model, input_indices, checkpoint


def compute_metrics_incremental(preds_list, targets_list, device):
    """Berechne Metriken inkrementell ohne alle Daten gleichzeitig im Speicher."""
    # Initialisiere Akkumulatoren
    total_mse = 0.0
    total_mae = 0.0
    total_pixels = 0
    
    # Für SAM
    total_angle_sum = 0.0
    total_angle_sq = 0.0
    total_angle_pixels = 0
    
    # Für band-spezifische Metriken
    band_mse_sum = None
    band_abs_sum = None
    num_bands = None
    
    for pred, target in zip(preds_list, targets_list):
        batch_size, C, H, W = pred.shape
        
        # Aktuelle Batch-Größe
        num_pixels = batch_size * H * W
        
        # MSE und MAE
        batch_mse = F.mse_loss(pred, target, reduction='sum').item()
        batch_mae = F.l1_loss(pred, target, reduction='sum').item()
        
        total_mse += batch_mse
        total_mae += batch_mae
        total_pixels += num_pixels
        
        # SAM für dieses Batch
        pred_flat = pred.view(batch_size, C, -1)  # (B, C, H*W)
        target_flat = target.view(batch_size, C, -1)
        
        pred_norm = F.normalize(pred_flat, dim=1)
        target_norm = F.normalize(target_flat, dim=1)
        
        cos_sim = (pred_norm * target_norm).sum(dim=1)  # (B, H*W)
        cos_sim = torch.clamp(cos_sim, -1, 1)
        
        angle = torch.acos(cos_sim) * 180 / np.pi  # Grad
        
        total_angle_sum += angle.sum().item()
        total_angle_sq += (angle ** 2).sum().item()
        total_angle_pixels += angle.numel()
        
        # Band-spezifische Metriken initialisieren
        if band_mse_sum is None:
            num_bands = C
            band_mse_sum = torch.zeros(C, device=device)
            band_abs_sum = torch.zeros(C, device=device)
        
        # Band-spezifische MSE und MAE
        for c in range(C):
            band_pred = pred[:, c, :, :]
            band_target = target[:, c, :, :]
            
            band_mse_sum[c] += F.mse_loss(band_pred, band_target, reduction='sum')
            band_abs_sum[c] += F.l1_loss(band_pred, band_target, reduction='sum')
        
        # Speicher freigeben
        del pred, target
        torch.cuda.empty_cache()
    
    # Finalisiere Metriken
    metrics = {}
    
    # Globale Metriken
    metrics['mse'] = total_mse / total_pixels if total_pixels > 0 else 0
    metrics['mae'] = total_mae / total_pixels if total_pixels > 0 else 0
    
    # PSNR
    if metrics['mse'] > 0:
        # Wir brauchen den maximalen Wert - sammle das auch inkrementell
        max_val = 0
        for target in targets_list:
            max_val = max(max_val, target.max().item())
        metrics['psnr'] = 10 * np.log10(max_val ** 2 / metrics['mse'])
    else:
        metrics['psnr'] = float('inf')
    
    # SAM
    if total_angle_pixels > 0:
        metrics['sam'] = total_angle_sum / total_angle_pixels
        angle_variance = (total_angle_sq / total_angle_pixels) - (metrics['sam'] ** 2)
        metrics['sam_std'] = np.sqrt(max(0, angle_variance))
    else:
        metrics['sam'] = 0.0
        metrics['sam_std'] = 0.0
    
    # Band-spezifische Metriken
    if band_mse_sum is not None:
        band_pixels_per_channel = total_pixels / num_bands if num_bands > 0 else 0
        metrics['rmse_per_band'] = torch.sqrt(band_mse_sum / band_pixels_per_channel).cpu().numpy()
        metrics['rel_error_per_band'] = (band_abs_sum / band_pixels_per_channel).cpu().numpy()
    
    return metrics


def evaluate_single_model(model, test_cube, input_indices, device, patch_size=128):
    """Evaluiere ein einzelnes Modell auf einem Test-Cube mit Speicher-Optimierung."""
    H, W, C = test_cube.shape
    
    # REDUZIERE Patch-Größe für weniger Speicherverbrauch
    patch_size = min(patch_size, 128)  # Maximal 128x128
    print(f"  Verwende Patch-Größe: {patch_size}x{patch_size}")
    
    # Schrittgröße erhöhen für weniger Overlap
    step_size = max(1, patch_size // 1)  # Kein Overlap für Speicherersparnis
    
    preds_list = []
    targets_list = []
    
    total_patches = ((H - patch_size + step_size) // step_size) * ((W - patch_size + step_size) // step_size)
    print(f"  Verarbeite ~{total_patches} Patches...")
    
    patch_count = 0
    for h_start in range(0, H - patch_size + 1, step_size):
        for w_start in range(0, W - patch_size + 1, step_size):
            patch_count += 1
            
            # Fortschrittsanzeige
            if patch_count % 50 == 0:
                print(f"  Verarbeitet {patch_count}/{total_patches} Patches...")
                torch.cuda.empty_cache()
            
            h_end = min(h_start + patch_size, H)
            w_end = min(w_start + patch_size, W)
            
            # Falls Patch zu klein am Rand, überspringen
            if h_end - h_start < patch_size // 2 or w_end - w_start < patch_size // 2:
                continue
            
            patch = test_cube[h_start:h_end, w_start:w_end, :]
            
            # Auf Patch-Größe auffüllen falls nötig
            if patch.shape[0] < patch_size or patch.shape[1] < patch_size:
                pad_h = patch_size - patch.shape[0]
                pad_w = patch_size - patch.shape[1]
                patch = np.pad(patch, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
            
            patch = patch.transpose(2, 0, 1)  # (C, H, W)
            
            # Zu Tensor konvertieren
            input_bands = torch.from_numpy(patch[input_indices]).unsqueeze(0).float().to(device)
            full_cube = torch.from_numpy(patch).unsqueeze(0).float().to(device)
            
            # Forward pass mit Speicher-Optimierung
            with torch.no_grad():
                with torch.cuda.amp.autocast():  # Mixed Precision für weniger Speicher
                    try:
                        S_pred, params = model(input_bands, full_cube)
                    except Exception as e:
                        # Fallback falls volle spektrale Information nicht benötigt wird
                        S_pred = model(input_bands)
            
            # Nur die tatsächliche Patch-Größe behalten (ohne Padding)
            actual_h = min(patch_size, h_end - h_start)
            actual_w = min(patch_size, w_end - w_start)
            
            if actual_h < patch_size or actual_w < patch_size:
                S_pred = S_pred[:, :, :actual_h, :actual_w]
                full_cube = full_cube[:, :, :actual_h, :actual_w]
            
            # Sofort CPU verschieben um GPU-Speicher freizugeben
            preds_list.append(S_pred.cpu())
            targets_list.append(full_cube.cpu())
            
            # Zwischenspeicher leeren
            del input_bands, full_cube, S_pred
            torch.cuda.empty_cache()
    
    print(f"  Alle {len(preds_list)} Patches verarbeitet")
    
    if not preds_list:
        raise ValueError("Keine gültigen Patches gefunden. Test-Cube zu klein oder Patch-Größe zu groß.")
    
    # Berechne Metriken inkrementell
    print("  Berechne Metriken...")
    metrics = compute_metrics_incremental(preds_list, targets_list, device)
    
    # Nur ein paar Beispiel-Patches behalten für Visualisierung
    max_examples = min(4, len(preds_list))
    example_preds = torch.cat(preds_list[:max_examples], dim=0).to(device)
    example_targets = torch.cat(targets_list[:max_examples], dim=0).to(device)
    
    return metrics, example_preds, example_targets


def plot_comparison(results_1band, results_5band, wavenumbers, save_path):
    """Erstelle Vergleichs-Plots."""
    try:
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig)  # Reduzierte Grid-Größe
        
        # 1. RMSE per Band Comparison
        ax1 = fig.add_subplot(gs[0, :2])
        if 'rmse_per_band' in results_1band['metrics'] and 'rmse_per_band' in results_5band['metrics']:
            ax1.plot(wavenumbers, results_1band['metrics']['rmse_per_band'], 
                     'b-', linewidth=2, label='1 Band Input')
            ax1.plot(wavenumbers, results_5band['metrics']['rmse_per_band'], 
                     'r-', linewidth=2, label='5 Band Input')
            ax1.set_xlabel('Wellenzahl (cm⁻¹)')
            ax1.set_ylabel('RMSE')
            ax1.set_title('RMSE pro Spektralband')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # 2. Relative Error per Band
        ax2 = fig.add_subplot(gs[0, 2:])
        if 'rel_error_per_band' in results_1band['metrics'] and 'rel_error_per_band' in results_5band['metrics']:
            ax2.plot(wavenumbers, results_1band['metrics']['rel_error_per_band'] * 100, 
                     'b-', linewidth=2, label='1 Band Input')
            ax2.plot(wavenumbers, results_5band['metrics']['rel_error_per_band'] * 100, 
                     'r-', linewidth=2, label='5 Band Input')
            ax2.set_xlabel('Wellenzahl (cm⁻¹)')
            ax2.set_ylabel('Relativer Fehler (%)')
            ax2.set_title('Relativer Fehler pro Spektralband')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Example Spectrum Comparison
        ax3 = fig.add_subplot(gs[1, :2])
        
        # Nimm erstes Patch, mittleres Pixel
        if results_1band['preds'].numel() > 0 and results_5band['preds'].numel() > 0:
            idx = 0
            h = min(32, results_1band['preds'].shape[2] - 1)
            w = min(32, results_1band['preds'].shape[3] - 1)
            
            target_spectrum = results_1band['targets'][idx, :, h, w].cpu().numpy()
            pred_1band_spectrum = results_1band['preds'][idx, :, h, w].cpu().numpy()
            pred_5band_spectrum = results_5band['preds'][idx, :, h, w].cpu().numpy()
            
            ax3.plot(wavenumbers, target_spectrum, 'k-', linewidth=2, label='Ground Truth')
            ax3.plot(wavenumbers, pred_1band_spectrum, 'b--', linewidth=2, label='1 Band')
            ax3.plot(wavenumbers, pred_5band_spectrum, 'r--', linewidth=2, label='5 Band')
            
            # Markiere Input-Bänder
            if 'input_indices' in results_1band:
                for idx_band in results_1band['input_indices']:
                    if idx_band < len(wavenumbers):
                        ax3.axvline(wavenumbers[idx_band], color='blue', linestyle=':', alpha=0.3)
            if 'input_indices' in results_5band:
                for idx_band in results_5band['input_indices']:
                    if idx_band < len(wavenumbers):
                        ax3.axvline(wavenumbers[idx_band], color='red', linestyle=':', alpha=0.3)
            
            ax3.set_xlabel('Wellenzahl (cm⁻¹)')
            ax3.set_ylabel('Strahlung')
            ax3.set_title('Spektraler Vergleich (Beispiel-Pixel)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Metrics Bar Chart
        ax4 = fig.add_subplot(gs[1, 2:])
        
        metrics_names = ['MSE (×1000)', 'MAE (×1000)', 'SAM (°)']
        if all(m in results_1band['metrics'] for m in ['mse', 'mae', 'sam']) and \
           all(m in results_5band['metrics'] for m in ['mse', 'mae', 'sam']):
            
            metrics_1band = [
                results_1band['metrics']['mse'] * 1000,
                results_1band['metrics']['mae'] * 1000,
                results_1band['metrics']['sam']
            ]
            metrics_5band = [
                results_5band['metrics']['mse'] * 1000,
                results_5band['metrics']['mae'] * 1000,
                results_5band['metrics']['sam']
            ]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            bars1 = ax4.bar(x - width/2, metrics_1band, width, label='1 Band', color='blue', alpha=0.7)
            bars2 = ax4.bar(x + width/2, metrics_5band, width, label='5 Band', color='red', alpha=0.7)
            
            ax4.set_ylabel('Wert')
            ax4.set_title('Metriken Vergleich')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics_names)
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis='y')
            
            # Werte über Balken
            for bar, val in zip(bars1, metrics_1band):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
            for bar, val in zip(bars2, metrics_5band):
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 5-6. Visual Comparison for different bands (nur 2 Bänder statt 4)
        bands_to_show = [0, min(35, len(wavenumbers)-1)]
        
        for i, band in enumerate(bands_to_show):
            row = 2
            col_start = i * 2
            
            ax_gt = fig.add_subplot(gs[row, col_start])
            ax_err = fig.add_subplot(gs[row, col_start + 1])
            
            # Ground Truth
            img_gt = results_1band['targets'][0, band].cpu().numpy()
            ax_gt.imshow(img_gt, cmap='hot', vmin=img_gt.min(), vmax=img_gt.max())
            ax_gt.set_title(f'GT Band {band} ({wavenumbers[band]:.0f} cm⁻¹)')
            ax_gt.axis('off')
            
            # Error Comparison
            err_1band = torch.abs(results_1band['preds'][0, band] - 
                                 results_1band['targets'][0, band]).cpu().numpy()
            err_5band = torch.abs(results_5band['preds'][0, band] - 
                                 results_5band['targets'][0, band]).cpu().numpy()
            
            # Side by side
            err_combined = np.concatenate([err_1band, err_5band], axis=1)
            im = ax_err.imshow(err_combined, cmap='viridis')
            ax_err.axvline(err_1band.shape[1], color='white', linewidth=2)
            ax_err.set_title(f'Error Band {band}: 1-Band | 5-Band')
            ax_err.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Vergleichs-Plot gespeichert: {save_path}")
    except Exception as e:
        print(f"Fehler beim Erstellen des Plots: {e}")
        import traceback
        traceback.print_exc()


def create_summary_report(results_1band, results_5band, wavenumbers, save_path):
    """Erstelle einen Text-Report."""
    try:
        report = []
        report.append("=" * 70)
        report.append("HADAR PINN Evaluation Report")
        report.append("=" * 70)
        report.append("")
        
        # Model Info
        report.append("MODELL-KONFIGURATION:")
        report.append("-" * 40)
        report.append(f"1-Band Modell:")
        if 'input_indices' in results_1band:
            indices = results_1band['input_indices']
            report.append(f"  Input-Bänder: {indices}")
            if len(indices) > 0 and indices[0] < len(wavenumbers):
                report.append(f"  Wellenzahlen: {wavenumbers[indices]}")
        report.append(f"5-Band Modell:")
        if 'input_indices' in results_5band:
            indices = results_5band['input_indices']
            report.append(f"  Input-Bänder: {indices}")
            if len(indices) > 0 and indices[0] < len(wavenumbers):
                report.append(f"  Wellenzahlen: {wavenumbers[indices]}")
        report.append("")
        
        # Metrics Comparison
        report.append("METRIKEN-VERGLEICH:")
        report.append("-" * 40)
        report.append(f"{'Metrik':<20} {'1-Band':<15} {'5-Band':<15} {'Verbesserung':<15}")
        report.append("-" * 65)
        
        for metric_name in ['mse', 'mae', 'psnr', 'sam']:
            if metric_name in results_1band['metrics'] and metric_name in results_5band['metrics']:
                val_1 = results_1band['metrics'][metric_name]
                val_5 = results_5band['metrics'][metric_name]
                
                if metric_name == 'psnr':
                    if val_1 != float('inf') and val_5 != float('inf'):
                        improvement = val_5 - val_1  # Höher ist besser
                        imp_str = f"+{improvement:.2f} dB"
                    else:
                        imp_str = "N/A"
                else:
                    if val_1 > 0:
                        improvement = (val_1 - val_5) / val_1 * 100  # Niedriger ist besser
                        imp_str = f"-{improvement:.1f}%"
                    else:
                        imp_str = "N/A"
                
                report.append(f"{metric_name.upper():<20} {val_1:<15.6f} {val_5:<15.6f} {imp_str:<15}")
        
        report.append("")
        
        # Band-wise Analysis
        if 'rmse_per_band' in results_1band['metrics'] and 'rmse_per_band' in results_5band['metrics']:
            report.append("BAND-ANALYSE (Top 5 schwierigste Bänder):")
            report.append("-" * 40)
            
            rmse_1band = results_1band['metrics']['rmse_per_band']
            rmse_5band = results_5band['metrics']['rmse_per_band']
            
            # Sortiere nach 1-Band RMSE
            sorted_indices = np.argsort(rmse_1band)[::-1][:5]
            
            report.append(f"{'Band':<10} {'Wellenzahl':<15} {'RMSE 1-Band':<15} {'RMSE 5-Band':<15}")
            report.append("-" * 55)
            
            for idx in sorted_indices:
                if idx < len(wavenumbers):
                    report.append(f"{idx:<10} {wavenumbers[idx]:<15.0f} {rmse_1band[idx]:<15.6f} {rmse_5band[idx]:<15.6f}")
            report.append("")
        
        # Conclusion
        report.append("FAZIT:")
        report.append("-" * 40)
        
        if 'rmse_per_band' in results_1band['metrics'] and 'rmse_per_band' in results_5band['metrics']:
            rmse_1band = results_1band['metrics']['rmse_per_band']
            rmse_5band = results_5band['metrics']['rmse_per_band']
            
            valid_mask = (rmse_1band > 0)
            if valid_mask.any():
                improvements = (rmse_1band[valid_mask] - rmse_5band[valid_mask]) / rmse_1band[valid_mask] * 100
                avg_improvement = np.mean(improvements)
                report.append(f"Durchschnittliche RMSE-Verbesserung: {avg_improvement:.1f}%")
        
        if 'sam' in results_1band['metrics'] and 'sam' in results_5band['metrics']:
            if results_5band['metrics']['sam'] < results_1band['metrics']['sam']:
                sam_imp = (results_1band['metrics']['sam'] - results_5band['metrics']['sam']) / results_1band['metrics']['sam'] * 100
                report.append(f"SAM-Verbesserung: {sam_imp:.1f}%")
                report.append(f"→ 5-Band Rekonstruktion zeigt bessere spektrale Genauigkeit")
        
        if 'psnr' in results_1band['metrics'] and 'psnr' in results_5band['metrics']:
            psnr_1 = results_1band['metrics']['psnr']
            psnr_5 = results_5band['metrics']['psnr']
            if psnr_1 != float('inf') and psnr_5 != float('inf'):
                psnr_diff = psnr_5 - psnr_1
                if psnr_diff > 0:
                    report.append(f"PSNR-Verbesserung: +{psnr_diff:.2f} dB")
        
        report.append("")
        report.append("=" * 70)
        
        # Speichern
        report_text = "\n".join(report)
        with open(save_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\nReport gespeichert: {save_path}")
    except Exception as e:
        print(f"Fehler beim Erstellen des Reports: {e}")


def main(args):
    # Setze CUDA environment variable um Speicherfragmentation zu vermeiden
    os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Speicher vor Training leeren
    torch.cuda.empty_cache()
    print(f"Initialer GPU-Speicher: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Lade Wellenzahlen und Materialbibliothek
    print("\nLade Daten...")
    try:
        wavenum_data = scio.loadmat(args.wavenumber_path)
        wavenumbers = wavenum_data['nu'].flatten().astype(np.float32)
        print(f"  Wellenzahlen: {len(wavenumbers)} Bänder")
    except Exception as e:
        print(f"Fehler beim Laden der Wellenzahlen: {e}")
        wavenumbers = np.linspace(720, 1250, 54).astype(np.float32)
        print(f"  Verwende Standard-Wellenzahlen: {len(wavenumbers)} Bänder")
    
    try:
        matlib_data = scio.loadmat(args.matlib_path)
        emissivity_lib = matlib_data['tmp'].astype(np.float32)
        print(f"  Materialbibliothek: {emissivity_lib.shape}")
    except Exception as e:
        print(f"Fehler beim Laden der Materialbibliothek: {e}")
        num_materials = 54
        num_bands = len(wavenumbers)
        emissivity_lib = np.random.randn(num_materials, num_bands).astype(np.float32)
        print(f"  Verwende zufällige Materialbibliothek: {emissivity_lib.shape}")
    
    # Lade Test-Cube
    print("\nLade Test-Cube...")
    try:
        test_data = scio.loadmat(args.test_cube)
        if 'S' in test_data:
            test_cube = test_data['S'].astype(np.float32)
        else:
            for key, value in test_data.items():
                if not key.startswith('__') and isinstance(value, np.ndarray) and value.ndim == 3:
                    test_cube = value.astype(np.float32)
                    print(f"  Verwende '{key}' als Test-Cube")
                    break
            else:
                raise ValueError("Kein geeignetes 3D-Array in der .mat Datei gefunden")
    except Exception as e:
        print(f"Fehler beim Laden des Test-Cubes: {e}")
        H, W = 540, 960  # HALBE Größe für weniger Speicherverbrauch
        C = len(wavenumbers)
        test_cube = np.random.randn(H, W, C).astype(np.float32)
        print(f"  Verwende zufälligen Test-Cube: {test_cube.shape}")
    
    print(f"Test Cube Shape: {test_cube.shape}")
    
    # Option: Nur einen Teil des Cubes verwenden für weniger Speicher
    if args.use_subset:
        print("Verwende nur Teil des Test-Cubes...")
        H, W, C = test_cube.shape
        subset_h = min(540, H)
        subset_w = min(960, W)
        test_cube = test_cube[:subset_h, :subset_w, :]
        print(f"  Reduzierte Größe: {test_cube.shape}")
    
    # Output Directory
    os.makedirs(args.output_path, exist_ok=True)
    
    results = {}
    
    # Evaluiere 1-Band Modell
    if args.model_1band and os.path.exists(args.model_1band):
        print("\nEvaluiere 1-Band Modell...")
        try:
            # Speicher leeren vor neuer Evaluation
            torch.cuda.empty_cache()
            
            model_1band, indices_1band, ckpt_1band = load_model(
                args.model_1band, wavenumbers, emissivity_lib, device
            )
            
            metrics_1band, preds_1band, targets_1band = evaluate_single_model(
                model_1band, test_cube, indices_1band, device, args.patch_size
            )
            
            results['1band'] = {
                'metrics': metrics_1band,
                'preds': preds_1band,
                'targets': targets_1band,
                'input_indices': indices_1band
            }
            
            print(f"  MSE: {metrics_1band['mse']:.6f}")
            print(f"  MAE: {metrics_1band['mae']:.6f}")
            print(f"  SAM: {metrics_1band['sam']:.2f}°")
            if 'psnr' in metrics_1band:
                print(f"  PSNR: {metrics_1band['psnr']:.2f} dB")
            
            # Speicher nach Evaluation leeren
            del model_1band
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Fehler bei der 1-Band Evaluation: {e}")
            import traceback
            traceback.print_exc()
    elif args.model_1band:
        print(f"Warnung: 1-Band Modell nicht gefunden: {args.model_1band}")
    
    # Evaluiere 5-Band Modell (falls vorhanden)
    if args.model_5band and os.path.exists(args.model_5band):
        print("\nEvaluiere 5-Band Modell...")
        try:
            # Speicher leeren vor neuer Evaluation
            torch.cuda.empty_cache()
            
            model_5band, indices_5band, ckpt_5band = load_model(
                args.model_5band, wavenumbers, emissivity_lib, device
            )
            
            metrics_5band, preds_5band, targets_5band = evaluate_single_model(
                model_5band, test_cube, indices_5band, device, args.patch_size
            )
            
            results['5band'] = {
                'metrics': metrics_5band,
                'preds': preds_5band,
                'targets': targets_5band,
                'input_indices': indices_5band
            }
            
            print(f"  MSE: {metrics_5band['mse']:.6f}")
            print(f"  MAE: {metrics_5band['mae']:.6f}")
            print(f"  SAM: {metrics_5band['sam']:.2f}°")
            if 'psnr' in metrics_5band:
                print(f"  PSNR: {metrics_5band['psnr']:.2f} dB")
            
            # Speicher nach Evaluation leeren
            del model_5band
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Fehler bei der 5-Band Evaluation: {e}")
            import traceback
            traceback.print_exc()
    elif args.model_5band:
        print(f"Warnung: 5-Band Modell nicht gefunden: {args.model_5band}")
    
    # Vergleichs-Analyse
    if '1band' in results and '5band' in results:
        print("\nErstelle Vergleichs-Analyse...")
        
        try:
            # Plots
            plot_comparison(
                results['1band'], 
                results['5band'], 
                wavenumbers,
                os.path.join(args.output_path, 'comparison_plot.png')
            )
        except Exception as e:
            print(f"Fehler beim Erstellen des Plots: {e}")
        
        try:
            # Report
            create_summary_report(
                results['1band'],
                results['5band'],
                wavenumbers,
                os.path.join(args.output_path, 'evaluation_report.txt')
            )
        except Exception as e:
            print(f"Fehler beim Erstellen des Reports: {e}")
    elif '1band' in results:
        print("\nNur 1-Band Modell evaluiert. Kein Vergleich möglich.")
        # Speichere trotzdem 1-Band Ergebnisse
        try:
            with open(os.path.join(args.output_path, '1band_results.txt'), 'w') as f:
                f.write(f"1-Band Modell Ergebnisse:\n")
                f.write(f"MSE: {results['1band']['metrics']['mse']:.6f}\n")
                f.write(f"MAE: {results['1band']['metrics']['mae']:.6f}\n")
                f.write(f"SAM: {results['1band']['metrics']['sam']:.2f}°\n")
                if 'psnr' in results['1band']['metrics']:
                    f.write(f"PSNR: {results['1band']['metrics']['psnr']:.2f} dB\n")
        except Exception as e:
            print(f"Fehler beim Speichern der 1-Band Ergebnisse: {e}")
    else:
        print("\nKeine Modell-Ergebnisse verfügbar.")
    
    # Speichere Ergebnisse
    try:
        save_dict = {}
        for model_key, model_results in results.items():
            for result_key, result_value in model_results.items():
                if result_key not in ['preds', 'targets']:
                    save_key = f"{model_key}_{result_key}"
                    
                    if torch.is_tensor(result_value):
                        save_dict[save_key] = result_value.cpu().numpy()
                    elif isinstance(result_value, dict):
                        for metric_key, metric_value in result_value.items():
                            metric_save_key = f"{model_key}_{result_key}_{metric_key}"
                            if torch.is_tensor(metric_value):
                                save_dict[metric_save_key] = metric_value.cpu().numpy()
                            else:
                                save_dict[metric_save_key] = metric_value
                    else:
                        save_dict[save_key] = result_value
        
        np.savez(
            os.path.join(args.output_path, 'evaluation_results.npz'),
            **save_dict
        )
        print(f"\nErgebnisse gespeichert in: {os.path.join(args.output_path, 'evaluation_results.npz')}")
    except Exception as e:
        print(f"Fehler beim Speichern der Ergebnisse: {e}")
    
    print(f"\nEvaluation abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HADAR PINN Evaluation')
    
    # Models
    parser.add_argument('--model_1band', type=str, default=None,
                        help='Pfad zum 1-Band Modell')
    parser.add_argument('--model_5band', type=str, default=None,
                        help='Pfad zum 5-Band Modell')
    
    # Data
    parser.add_argument('--test_cube', type=str, required=True,
                        help='Pfad zum Test Heat Cube')
    parser.add_argument('--wavenumber_path', type=str, required=True,
                        help='Pfad zur Wellenzahlen-Datei')
    parser.add_argument('--matlib_path', type=str, required=True,
                        help='Pfad zur Materialbibliothek')
    
    # Settings
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Patch Size für Evaluation (kleiner = weniger Speicher)')
    parser.add_argument('--output_path', type=str, default='./evaluation_results',
                        help='Output-Pfad')
    parser.add_argument('--use_subset', action='store_true',
                        help='Nur Teil des Test-Cubes verwenden')
    
    args = parser.parse_args()
    main(args)