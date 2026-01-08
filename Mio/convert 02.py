# convert.py
# HSI -> RGB mit robusten colour-science Fallbacks (ohne sRGB_COLOURSPACE)

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d

# --- colour-science: nur das Nötigste, robust gegen Versionen ---
try:
    from colour import MSDS_CMFS, SDS_ILLUMINANTS
except ImportError:
    from colour.colorimetry import MSDS_CMFS, SDS_ILLUMINANTS

from colour.models import XYZ_to_RGB

# ==== Konfiguration ====
USE_SUPERRES = False        # True = mehr Bänder (kann weicher wirken)
TARGET_BANDS = 200
PATCH_RADIUS = 3            # 2–4
TARGET_P99 = 0.85           # 0.8 kontrastreicher, 0.9 heller
INPUT_MAT = "./data/imgb9.mat"
OUTPUT_DIR = "./outputs/test_rgb"

# sRGB (D65) Konstanten – unabhängig von colour-Version nutzbar
SRGB_WHITEPOINT_XY = np.array([0.3127, 0.3290], dtype=np.float32)
SRGB_MATRIX_XYZ_TO_RGB = np.array(
    [[ 3.2404542, -1.5371385, -0.4985314],
     [-0.9692660,  1.8760108,  0.0415560],
     [ 0.0556434, -0.2040259,  1.0572252]],
    dtype=np.float32
)

# sRGB OETF (linear -> display)
try:
    from colour.models import eotf_inverse_sRGB  # bevorzugt
except Exception:
    def eotf_inverse_sRGB(x):
        x = np.clip(x, 0.0, 1.0)
        a = 0.055
        return np.where(x <= 0.0031308, 12.92 * x, (1 + a) * np.power(x, 1 / 2.4) - a)


# ===== Loader: v7 + v7.3 =====
def load_hyper_mat(mat_path, prefer_keys=("ref", "hcube", "cube", "data", "H")):
    """
    Lädt Hyperspektral-Daten aus .mat:
      - v7 (scipy) oder v7.3/HDF5 (h5py)
      - liefert (H, W, B) in float32
    """
    arr = None
    try:
        md = scipy.io.loadmat(mat_path, simplify_cells=True)
        for k in prefer_keys:
            if k in md:
                arr = np.asarray(md[k])
                break
    except NotImplementedError:
        pass

    if arr is None:
        import h5py
        with h5py.File(mat_path, "r") as f:
            for k in prefer_keys:
                if k in f:
                    arr = np.array(f[k])
                    break
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

    arr = np.asarray(arr).astype(np.float32)
    if arr.ndim != 3:
        raise ValueError(f"Erwarte 3D-Hypercube, bekam Shape {arr.shape}")

    # Ziel: (H, W, B)
    H, W, B = arr.shape
    common_band_counts = {16, 20, 25, 31, 51, 61, 81, 93, 101, 121, 151, 181, 200, 224, 240}
    if H in common_band_counts and H < W and H < B:
        arr = np.transpose(arr, (1, 2, 0))  # (B,H,W) -> (H,W,B)
    elif W in common_band_counts and W < H and W < B:
        arr = np.transpose(arr, (0, 2, 1))  # (H,B,W) -> (H,W,B)
    return arr


# ===== Superresolution (optional) =====
def enhance_spectral_resolution(hcube, target_bands=200):
    H, W, B = hcube.shape
    x_old = np.linspace(0, 1, B, dtype=np.float32)
    x_new = np.linspace(0, 1, target_bands, dtype=np.float32)
    out = np.empty((H, W, target_bands), dtype=np.float32)
    for i in range(H):
        f = interp1d(x_old, hcube[i, :, :], kind='cubic', axis=-1,
                     fill_value='extrapolate', assume_sorted=True)
        row = f(x_new)
        # Clamping gegen Overshoot:
        smin = np.min(hcube[i, :, :], axis=-1, keepdims=True)
        smax = np.max(hcube[i, :, :], axis=-1, keepdims=True)
        row = np.clip(row, smin, smax)
        out[i] = row
    return out


# ===== Wellenlängenschätzung =====
def estimate_wavelengths(hcube):
    mean_spectrum = np.mean(hcube, axis=(0, 1))
    n = len(mean_spectrum)
    win = max(5, min(11, n - (1 - n % 2)))
    if win % 2 == 0:
        win -= 1
    smoothed = savgol_filter(mean_spectrum, win, 3)
    wl = np.linspace(380, 780, n)
    # kleiner adaptiver Shift
    peak_region = max(8, n // 3)
    shift = int(np.clip(np.argmax(smoothed[:peak_region]) - peak_region * 0.5, -20, 20))
    return np.clip(wl - shift, 380, 780)


# ===== Neutraler Patch (hell + wenig Varianz) =====
def find_neutral_patch(hcube_vis, patch_radius=3):
    H, W, B = hcube_vis.shape
    mean_img = np.mean(hcube_vis, axis=2)
    std_img = np.std(hcube_vis, axis=2)

    def norm01(img):
        a, b = np.percentile(img, 1), np.percentile(img, 99)
        if b > a:
            img = (img - a) / (b - a)
        return np.clip(img, 0, 1)

    m = norm01(mean_img)          # hoch gut
    s = 1.0 - norm01(std_img)     # niedrig gut
    score = 0.7 * m + 0.3 * s

    r = patch_radius
    score[:r, :] = -np.inf
    score[-r:, :] = -np.inf
    score[:, :r] = -np.inf
    score[:, -r:] = -np.inf

    y, x = np.unravel_index(np.argmax(score), score.shape)
    return int(x), int(y)


def check_calibration(hcube, ref_x, ref_y, tol=0.1, r=3):
    patch = hcube[ref_y-r:ref_y+r+1, ref_x-r:ref_x+r+1, :]
    patch_mean = np.mean(patch, axis=(0, 1))
    below = patch_mean < (0.95 - tol)
    above = patch_mean > (0.95 + tol)
    if np.any(below) or np.any(above):
        print("[⚠️] Weißreferenz außerhalb Toleranzbereich!")
        print("    Min Abw.:", float(np.min(patch_mean)))
        print("    Max Abw.:", float(np.max(patch_mean)))
    else:
        print("[✅] Weißreferenz innerhalb tolerierbarer Grenzen")
    return patch_mean


# ===== Integration -> XYZ (Δλ-Gewichtung) =====
def integrate_to_xyz(hcube_norm, wavelengths_vis):
    cmfs = MSDS_CMFS['CIE 1931 2 Degree Standard Observer']
    illuminant = SDS_ILLUMINANTS['D65']

    wl_cmfs = np.asarray(cmfs.wavelengths)
    cmfs_vals = np.asarray(cmfs.values)  # (N,3)
    valid = (wl_cmfs >= 380) & (wl_cmfs <= 780)

    wl = wavelengths_vis.astype(np.float32)
    xbar = np.interp(wl, wl_cmfs[valid], cmfs_vals[valid, 0])
    ybar = np.interp(wl, wl_cmfs[valid], cmfs_vals[valid, 1])
    zbar = np.interp(wl, wl_cmfs[valid], cmfs_vals[valid, 2])

    ill_wl = np.asarray(illuminant.wavelengths)
    ill_vals = np.asarray(illuminant.values)
    e = np.interp(wl, ill_wl, ill_vals)

    dl = np.gradient(wl)
    W = np.stack([xbar * e * dl, ybar * e * dl, zbar * e * dl], axis=0)  # (3,B)

    xyz = np.tensordot(hcube_norm, W.T, axes=([2], [0]))  # (H,W,3)
    denom = (np.max(xyz[..., 1]) + 1e-8)
    if denom > 0:
        xyz = xyz / denom
    return xyz


# ===== XYZ -> sRGB (linear) + Belichtungs-Tonemapping =====
def xyz_to_srgb_linear(xyz):
    """
    XYZ -> lineares sRGB mit fixer sRGB-Matrix, D65 -> D65 (keine Adaption).
    """
    rgb_lin = XYZ_to_RGB(
        xyz,
        SRGB_WHITEPOINT_XY,  # XYZ-Whitepoint (xy)
        SRGB_WHITEPOINT_XY,  # RGB-Whitepoint (xy)
        SRGB_MATRIX_XYZ_TO_RGB,
    )
    return rgb_lin


def exposure_tonemap_srgb(rgb_lin, target_p99=0.85):
    """
    Belichtungs-Tonemapping: skaliert lineares sRGB so,
    dass das 99. Perzentil der Luminanz ~ target_p99 ist, dann sRGB-OETF.
    """
    Y = 0.2126 * rgb_lin[..., 0] + 0.7152 * rgb_lin[..., 1] + 0.0722 * rgb_lin[..., 2]
    p99 = np.percentile(Y, 99.0)
    gain = (target_p99 / (p99 + 1e-8)) if p99 > 0 else 1.0
    rgb_lin = np.clip(rgb_lin * gain, 0.0, 1.0)
    rgb_srgb = eotf_inverse_sRGB(rgb_lin)
    rgb_srgb = np.clip(rgb_srgb, 0.0, 1.0)
    return rgb_srgb


# ===== Hauptpipeline =====
def convert_hsi_to_rgb(mat_path, output_path,
                       use_superres=USE_SUPERRES, target_bands=TARGET_BANDS,
                       patch_radius=PATCH_RADIUS, target_p99=TARGET_P99):

    # 1) Laden
    hcube = load_hyper_mat(mat_path)  # (H,W,B)
    hcube = np.clip(hcube, 0, None)

    # 2) Grob-Normierung (gegen Ausreißer)
    p = np.percentile(hcube, 99.99)
    if p > 0:
        hcube = hcube / p

    # 3) Optional Super-Resolution
    if use_superres:
        print(f"[🔁] Spektrale Superresolution → {target_bands} Bänder ...")
        hcube = enhance_spectral_resolution(hcube, target_bands=target_bands)

    # 4) sichtbarer Bereich
    wavelengths = estimate_wavelengths(hcube)
    mask = (wavelengths >= 380) & (wavelengths <= 780)
    hcube_vis = hcube[:, :, mask]
    wl_vis = wavelengths[mask]

    # 5) neutralen Patch finden
    ref_x, ref_y = find_neutral_patch(hcube_vis, patch_radius=patch_radius)
    r = int(patch_radius)
    ref_x = int(np.clip(ref_x, r, hcube_vis.shape[1] - r - 1))
    ref_y = int(np.clip(ref_y, r, hcube_vis.shape[0] - r - 1))
    print(f"[🎯] Neutraler Patch bei: ({ref_x}, {ref_y})")

    # 6) Weißreferenz + Normalisierung
    check_calibration(hcube_vis, ref_x, ref_y, r=r)
    patch = hcube_vis[ref_y-r:ref_y+r+1, ref_x-r:ref_x+r+1, :]
    white_ref = np.mean(patch, axis=(0, 1))
    white_ref = np.clip(white_ref, 1e-6, None)
    hcube_norm = hcube_vis / white_ref

    # 7) Integration -> XYZ
    xyz = integrate_to_xyz(hcube_norm, wl_vis)

    # 8) XYZ -> lineares sRGB -> Tonemapping + sRGB-OETF
    rgb_lin = xyz_to_srgb_linear(xyz)
    rgb = exposure_tonemap_srgb(rgb_lin, target_p99=target_p99)

    # 9) Speichern
    os.makedirs(output_path, exist_ok=True)
    filename = os.path.splitext(os.path.basename(mat_path))[0]
    save_path = os.path.join(output_path, f"{filename}_02_rgb.png")
    plt.imsave(save_path, rgb)
    print(f"[✅] RGB gespeichert unter: {save_path}")


if __name__ == "__main__":
    convert_hsi_to_rgb(
        mat_path=INPUT_MAT,
        output_path=OUTPUT_DIR,
        use_superres=USE_SUPERRES,
        target_bands=TARGET_BANDS,
        patch_radius=PATCH_RADIUS,
        target_p99=TARGET_P99,
    )
