# convert.py
# Vollständiges Skript: Laden (v7/v7.3), HSI->RGB, robustere colour-Imports, Guards.

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# ---- robustere colour-science Imports ----
try:
    from colour import MSDS_CMFS, SDS_ILLUMINANTS
    from colour.models import XYZ_to_sRGB
except ImportError:
    from colour.colorimetry import MSDS_CMFS, SDS_ILLUMINANTS
    from colour.models import XYZ_to_sRGB


# === Loader, der v7 + v7.3 kann ===
def load_hyper_mat(mat_path, prefer_keys=("ref", "hcube", "cube", "data", "H")):
    """
    Lädt eine Hyperspektral-MAT-Datei:
    - v7 (scipy) oder v7.3/HDF5 (h5py)
    - findet gängige Keys (z.B. 'ref')
    - liefert (H, W, B) als float32
    """
    arr = None
    # 1) Versuch: klassisches MAT (<= v7)
    try:
        md = scipy.io.loadmat(mat_path, simplify_cells=True)
        for k in prefer_keys:
            if k in md:
                arr = np.asarray(md[k])
                break
    except NotImplementedError:
        # v7.3, HDF5-basiert
        pass

    # 2) HDF5-Pfad (v7.3)
    if arr is None:
        import h5py
        with h5py.File(mat_path, "r") as f:
            # bevorzugte Keys
            for k in prefer_keys:
                if k in f:
                    arr = np.array(f[k])
                    break
            # erster Dataset, falls nichts passt
            if arr is None:
                def first_dataset(g):
                    for name in g:
                        obj = g[name]
                        if isinstance(obj, h5py.Dataset):
                            return np.array(obj)
                        if isinstance(obj, h5py.Group):
                            x = first_dataset(obj)
                            if x is not None:
                                return x
                    return None
                arr = first_dataset(f)
            if arr is None:
                raise KeyError("Kein geeignetes Dataset in der MAT/HDF5-Datei gefunden.")

    # 3) In float32 und Achsen prüfen
    arr = np.asarray(arr).astype(np.float32)

    if arr.ndim != 3:
        raise ValueError(f"Erwarte 3D-Hypercube, bekam Shape {arr.shape}")

    H, W, B = arr.shape
    # Heuristik für (B,H,W) etc. -> Ziel (H,W,B)
    common_band_counts = {16, 20, 25, 31, 51, 61, 81, 93, 101, 121, 151, 181, 200, 224, 240}
    if H in common_band_counts and H < W and H < B:
        arr = np.transpose(arr, (1, 2, 0))  # (B,H,W) -> (H,W,B)
    elif W in common_band_counts and W < H and W < B:
        arr = np.transpose(arr, (0, 2, 1))  # (H,B,W) -> (H,W,B)
    # sonst: wir nehmen an, es ist schon (H,W,B)

    return arr


# === 🔁 Superresolution: mehr Kanäle via Interpolation ===
def enhance_spectral_resolution(hcube, target_bands=200):
    """Interpoliert spektrale Achse auf Zielanzahl Bänder."""
    H, W, B = hcube.shape
    hcube_interp = np.zeros((H, W, target_bands), dtype=np.float32)
    x_old = np.linspace(0, 1, B)
    x_new = np.linspace(0, 1, target_bands)

    # Vektorisiert wäre schneller, aber klarer so:
    for i in range(H):
        for j in range(W):
            f = interp1d(x_old, hcube[i, j, :], kind='cubic', fill_value='extrapolate', assume_sorted=True)
            hcube_interp[i, j, :] = f(x_new)
    return hcube_interp


# === 🎯 Neutrales Referenzfeld finden ===
def find_neutral_patch(hcube):
    """Findet Position mit geringster spektraler Varianz (graue Fläche)."""
    spectral_std = np.std(hcube, axis=2)
    y, x = np.unravel_index(np.argmin(spectral_std), spectral_std.shape)
    return x, y


# === ✅ Kalibrierung prüfen ===
def check_calibration(hcube, ref_x, ref_y, tol=0.1):
    """Warnung, wenn der Referenzbereich nicht ca. 0.95 +/- tol ist."""
    patch = hcube[ref_y-2:ref_y+3, ref_x-2:ref_x+3, :]
    patch_mean = np.mean(patch, axis=(0, 1))
    below = patch_mean < (0.95 - tol)
    above = patch_mean > (0.95 + tol)

    if np.any(below) or np.any(above):
        print("[⚠️] Warnung: Weißreferenz außerhalb Toleranzbereich!")
        print("    Min Abw.:", float(np.min(patch_mean)))
        print("    Max Abw.:", float(np.max(patch_mean)))
    else:
        print("[✅] Weißreferenz innerhalb tolerierbarer Grenzen")
    return patch_mean


# === 🔍 Wellenlängenschätzung ===
def estimate_wavelengths(hcube):
    """Schätzt Wellenlängenpositionen aus dem mittleren Verlauf (Fallback 380–780 nm)."""
    mean_spectrum = np.mean(hcube, axis=(0, 1))
    # Savitzky-Golay: Fenster robust wählen (ungerade, >=5)
    win = min(11, len(mean_spectrum) - (1 - len(mean_spectrum) % 2))
    win = max(5, win)  # mindestens 5
    if win % 2 == 0:
        win += 1
    win = min(win, len(mean_spectrum) - (1 - len(mean_spectrum) % 2))
    smoothed = savgol_filter(mean_spectrum, win, 3)
    estimated = np.linspace(400, 700, len(mean_spectrum))
    peak_shift = int(np.argmax(smoothed[:len(mean_spectrum)//3]) - 50)
    return np.clip(estimated - peak_shift, 380, 780)


# === 🌈 Hyperspektral zu RGB ===
def convert_hsi_to_rgb(mat_path, output_path, use_superres=False, target_bands=200):
    # 1) Laden
    hcube = load_hyper_mat(mat_path)  # (H, W, B), float32
    hcube = np.clip(hcube, 0, None)

    # 2) Grob-Normierung (gegen Ausreißer)
    perc = np.percentile(hcube, 99.99)
    if perc > 0:
        hcube = hcube / perc

    # 3) Optional Super-Resolution
    if use_superres:
        print(f"[🔁] Spektrale Superresolution → {target_bands} Bänder ...")
        hcube = enhance_spectral_resolution(hcube, target_bands=target_bands)

    # 4) Sichtbarer Bereich schätzen & filtern
    wavelengths = estimate_wavelengths(hcube)
    visible_mask = (wavelengths >= 380) & (wavelengths <= 780)
    hcube_vis = hcube[:, :, visible_mask]
    wavelengths_vis = wavelengths[visible_mask]

    # 5) Grauen Referenz-Patch finden + begrenzen (Rand-Guard)
    ref_x, ref_y = find_neutral_patch(hcube_vis)
    ref_x = int(np.clip(ref_x, 2, hcube_vis.shape[1] - 3))
    ref_y = int(np.clip(ref_y, 2, hcube_vis.shape[0] - 3))
    print(f"[🎯] Neutraler Patch bei: ({ref_x}, {ref_y})")

    # 6) Kalibrierung checken & Weißnormalisierung
    check_calibration(hcube_vis, ref_x, ref_y)
    patch = hcube_vis[ref_y-2:ref_y+3, ref_x-2:ref_x+3, :]
    white_ref = np.mean(patch, axis=(0, 1))
    hcube_norm = hcube_vis / (white_ref + 1e-12)

    # 7) XYZ-Gewichte aus CMFs & Illuminant D65
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = SDS_ILLUMINANTS['D65']

    wl_cmfs = np.asarray(cmfs.wavelengths)
    cmfs_vals = np.asarray(cmfs.values)  # shape: (N_wl, 3)
    valid_cmfs = (wl_cmfs >= 380) & (wl_cmfs <= 780)

    # Interpolierte x̄, ȳ, z̄ für unsere Wellenlängen
    xyz_weights = np.array([
        np.interp(wavelengths_vis, wl_cmfs[valid_cmfs], cmfs_vals[valid_cmfs, i])
        for i in range(3)
    ])  # shape: (3, B_vis)

    # Illuminant auf unsere Wellenlängen mappen
    ill_wl = np.asarray(illuminant.wavelengths)
    ill_vals = np.asarray(illuminant.values)
    ill_interp = np.interp(wavelengths_vis, ill_wl, ill_vals)

    # Kombination
    xyz_weights *= ill_interp  # (3, B_vis)

    # 8) Spektrale Faltung → XYZ
    # hcube_norm: (H, W, B_vis), xyz_weights.T: (B_vis, 3)
    xyz = np.tensordot(hcube_norm, xyz_weights.T, axes=([2], [0]))  # (H, W, 3)

    # Weiß-Normierung gegen Y
    denom = (xyz[..., 1].max() + 1e-8)
    if denom > 0:
        xyz = xyz / denom

    # 9) XYZ → sRGB (linear→gamma handled by colour)
    rgb = XYZ_to_sRGB(xyz)
    # leichte Aufhellung/Kompression
    rgb = np.clip(np.sqrt(np.clip(rgb, 0, 1)), 0, 1)

    # 10) Speichern
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.splitext(os.path.basename(mat_path))[0]
    save_path = os.path.join(output_path, f"{filename}_rgb.png")
    plt.imsave(save_path, rgb)
    print(f"[✅] RGB gespeichert unter: {save_path}")


if __name__ == "__main__":
    # ---- Pfade anpassen ----
    convert_hsi_to_rgb(
        mat_path="./data/img1.mat",
        output_path="./outputs/test_rgb",
        use_superres=True,      # oder False
        target_bands=200
    )
