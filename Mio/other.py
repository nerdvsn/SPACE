import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import io, color

def analyze_rgb_image(image_path):
    if not os.path.exists(image_path):
        print(f"[❌] Bild nicht gefunden: {image_path}")
        return

    # Bild laden
    rgb = io.imread(image_path)
    rgb = rgb.astype(np.float32) / 255 if rgb.max() > 1 else rgb

    print(f"[INFO] Bildform: {rgb.shape}, Wertebereich: {rgb.min():.4f} – {rgb.max():.4f}")

    # Alphakanal entfernen
    if rgb.shape[-1] == 4:
        print("[ℹ️] Alphakanal erkannt – wird entfernt")
        rgb = rgb[..., :3]

    # RGB Bild anzeigen
    plt.figure(figsize=(6, 6))
    plt.imshow(rgb)
    plt.title("Tesla RGB – Visualisierung")
    plt.axis("off")
    plt.show()

    try:
        hsv = color.rgb2hsv(rgb)
        luminance = color.rgb2gray(rgb)

        stats = {
            "Mean RGB": np.mean(rgb, axis=(0, 1)),
            "Contrast (Y)": float(np.std(luminance)),
            "Brightness (Y)": float(np.mean(luminance)),
            "Saturation (avg)": float(np.mean(hsv[..., 1])),
            "Dynamic range (Y)": float(np.max(luminance) - np.min(luminance))
        }

        print("\n📊 Bildmetriken:")
        for k, v in stats.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: {np.round(v, 4)}")
            else:
                print(f"  {k}: {v:.4f}")

        # Histogramm der Luminanz
        plt.figure()
        plt.hist(luminance.ravel(), bins=256, color='gray', alpha=0.8)
        plt.title("Histogramm – Luminanz (Y-Kanal)")
        plt.xlabel("Helligkeit")
        plt.ylabel("Anzahl Pixel")
        plt.grid(True)
        plt.show()

    except Exception as e:
        print(f"[⚠️] Fehler bei der Analyse: {e}")

# 🔧 Anwendung
if __name__ == "__main__":
    image_path = "./data/imgf1.png"  # Anpassen, falls Pfad anders
    analyze_rgb_image(image_path)
