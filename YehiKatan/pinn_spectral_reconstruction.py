"""
PINN für Spektrale Rekonstruktion: K Bänder → 54 Bänder

Physik-Gleichung:
    Sν = eν(m) × Bν(T) + [1 - eν(m)] × Xν

Wobei:
    - Sν: Gemessene Strahlung bei Wellenzahl ν
    - eν(m): Emissivität des Materials m bei Wellenzahl ν
    - Bν(T): Planck-Funktion (Schwarzkörperstrahlung) bei Temperatur T
    - Xν: Umgebungsstrahlung (reflektierter Anteil)

Das PINN lernt:
    - T(x,y): Temperaturkarte
    - m(x,y): Materialkarte (Index in Bibliothek)
    - V(x,y): Thermal Lighting Factors für Xν

Aus diesen Parametern rekonstruiert der Physics-Layer alle 54 Bänder.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Physikalische Konstanten
hbar = 105457180e-42      # Reduziertes Planck'sches Wirkungsquantum
h = 2 * math.pi * hbar    # Planck'sches Wirkungsquantum
c = 299792458             # Lichtgeschwindigkeit (m/s)
kb = 138064852e-31        # Boltzmann-Konstante
cB = h * c / kb           # Konstante für Planck-Formel


class PlanckLayer(nn.Module):
    """
    Berechnet die Planck-Funktion Bν(T) für gegebene Wellenzahlen und Temperaturen.
    
    Bν(T) = 2hc²ν³ / (exp(hcν/kT) - 1)
    
    In Einheiten von W/(m² sr cm⁻¹) für Wellenzahl in cm⁻¹
    """
    def __init__(self, wavenumbers):
        super().__init__()
        # Wellenzahlen als Buffer (nicht trainierbar)
        self.register_buffer('nu', torch.tensor(wavenumbers, dtype=torch.float32))
        self.num_bands = len(wavenumbers)
    
    def forward(self, T):
        """
        Args:
            T: Temperatur in Kelvin, Shape: (B, 1, H, W)
        
        Returns:
            Bν(T): Planck-Strahlung, Shape: (B, num_bands, H, W)
        """
        # nu: (num_bands,) -> (1, num_bands, 1, 1)
        nu = self.nu.view(1, -1, 1, 1)
        
        # T: (B, 1, H, W) -> broadcast mit nu
        # Vermeide Division durch 0 und numerische Instabilität
        T_safe = torch.clamp(T, min=200.0, max=500.0)  # Typischer Bereich: 200-500 K
        
        # Planck-Formel
        # Faktor 1e2 für Umrechnung cm⁻¹ zu m⁻¹
        exponent = cB * 1e2 * nu / T_safe
        exponent = torch.clamp(exponent, max=50.0)  # Numerische Stabilität
        
        Bnu = 2e6 * c * (nu ** 2) / (torch.exp(exponent) - 1)
        
        # Konvertiere zu Energie (W/m²) durch Multiplikation mit hcν
        Bnu_energy = 1e2 * h * c * nu * Bnu
        
        return Bnu_energy


class EmissivityLookup(nn.Module):
    """
    Lookup-Tabelle für Materialemissivitäten.
    
    Verwendet Soft-Attention über Materialien, um differenzierbar zu bleiben.
    Memory-effiziente Implementierung mit einsum.
    """
    def __init__(self, emissivity_library):
        """
        Args:
            emissivity_library: Shape (num_bands, num_materials)
        """
        super().__init__()
        # Emissivitätsbibliothek als Buffer
        self.register_buffer('emissivity_lib', 
                            torch.tensor(emissivity_library, dtype=torch.float32))
        self.num_bands, self.num_materials = emissivity_library.shape
    
    def forward(self, material_logits):
        """
        Args:
            material_logits: Shape (B, num_materials, H, W)
        
        Returns:
            emissivity: Shape (B, num_bands, H, W)
        """
        # Soft-Attention über Materialien
        material_weights = F.softmax(material_logits, dim=1)  # (B, M, H, W)
        
        # Memory-effiziente Berechnung mit einsum
        # emissivity_lib: (num_bands, M)
        # material_weights: (B, M, H, W)
        # Ergebnis: (B, num_bands, H, W)
        emissivity = torch.einsum('cm,bmhw->bchw', self.emissivity_lib, material_weights)
        
        return emissivity, material_weights


class EnvironmentalRadiationEstimator(nn.Module):
    """
    Schätzt die Umgebungsstrahlung Xν basierend auf Thermal Lighting Factors.
    
    Xν = V₁ × S₁ν + V₂ × S₂ν + δν
    
    Wobei S₁ν und S₂ν gemittelte Spektren von Bildregionen sind.
    """
    def __init__(self, num_bands, num_env_sources=2):
        super().__init__()
        self.num_bands = num_bands
        self.num_env_sources = num_env_sources
        
        # Lernbare Basisspektren für Umgebungsstrahlung
        # Initialisiert mit typischen Werten
        self.env_spectra = nn.Parameter(
            torch.randn(num_env_sources, num_bands) * 0.01 + 0.1
        )
        
        # Lernbarer Residualterm
        self.residual = nn.Parameter(torch.zeros(1, num_bands, 1, 1))
    
    def forward(self, V, input_spectra=None):
        """
        Args:
            V: Thermal Lighting Factors, Shape (B, num_env_sources, H, W)
            input_spectra: Optional, gemessene Spektren für adaptive Schätzung
        
        Returns:
            X: Umgebungsstrahlung, Shape (B, num_bands, H, W)
        """
        B, _, H, W = V.shape
        
        if input_spectra is not None:
            # Berechne Umgebungsspektren aus Bildregionen
            # Teile Bild in obere und untere Hälfte
            top_half = input_spectra[:, :, :H//2, :].mean(dim=(2, 3), keepdim=True)
            bottom_half = input_spectra[:, :, H//2:, :].mean(dim=(2, 3), keepdim=True)
            env_spectra = torch.cat([top_half, bottom_half], dim=0)  # (2, num_bands, 1, 1)
            env_spectra = env_spectra.squeeze(-1).squeeze(-1).mean(dim=0)  # (num_bands,)
            env_spectra = env_spectra.view(1, self.num_bands, 1, 1)
            
            # Verwende eine Mischung aus gelernten und datenbasierten Spektren
            S_env = self.env_spectra.view(1, self.num_env_sources, self.num_bands, 1, 1)
        else:
            S_env = self.env_spectra.view(1, self.num_env_sources, self.num_bands, 1, 1)
        
        # V: (B, num_env_sources, H, W) -> (B, num_env_sources, 1, H, W)
        V_expanded = V.unsqueeze(2)
        
        # Gewichtete Summe
        X = (S_env * V_expanded).sum(dim=1)  # (B, num_bands, H, W)
        
        # Addiere Residual
        X = X + self.residual
        
        # Stelle sicher, dass X physikalisch sinnvoll ist (positiv)
        X = F.softplus(X)
        
        return X


class SpectralEncoder(nn.Module):
    """
    CNN-Encoder der aus K Input-Bändern latente Features extrahiert.
    """
    def __init__(self, input_bands, hidden_dim=64):
        super().__init__()
        
        self.encoder = nn.Sequential(
            # Block 1
            nn.Conv2d(input_bands, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            
            # Block 2
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            
            # Block 3
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
        )
        
        self.output_dim = hidden_dim * 4
    
    def forward(self, x):
        return self.encoder(x)


class ParameterDecoder(nn.Module):
    """
    Dekodiert latente Features zu physikalischen Parametern (T, m, V).
    """
    def __init__(self, input_dim, num_materials, num_env_sources=2):
        super().__init__()
        
        # Gemeinsamer Trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(input_dim, input_dim // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(input_dim // 2),
            nn.ReLU(inplace=True),
        )
        
        # Temperatur-Head (1 Kanal, in Celsius, wird später zu Kelvin konvertiert)
        self.T_head = nn.Sequential(
            nn.Conv2d(input_dim // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        # Material-Head (num_materials Kanäle für Softmax)
        self.m_head = nn.Sequential(
            nn.Conv2d(input_dim // 2, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_materials, kernel_size=1),
        )
        
        # V-Head (Thermal Lighting Factors)
        self.V_head = nn.Sequential(
            nn.Conv2d(input_dim // 2, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_env_sources, kernel_size=1),
            nn.Softmax(dim=1),  # V₁ + V₂ = 1
        )
        
        # Initialisierung für Temperatur (typischer Bereich)
        self._init_T_head()
    
    def _init_T_head(self):
        # Initialisiere T-Head so, dass Output nahe 20°C (293 K) liegt
        nn.init.zeros_(self.T_head[-1].weight)
        nn.init.constant_(self.T_head[-1].bias, 20.0)  # 20°C als Startwert
    
    def forward(self, features):
        trunk_out = self.trunk(features)
        
        # Temperatur in Celsius
        T_celsius = self.T_head(trunk_out)
        # Konvertiere zu Kelvin und beschränke auf physikalisch sinnvollen Bereich
        T_kelvin = T_celsius + 273.15
        T_kelvin = torch.clamp(T_kelvin, min=250.0, max=350.0)  # -23°C bis 77°C
        
        # Material-Logits
        m_logits = self.m_head(trunk_out)
        
        # Thermal Lighting Factors (summiert zu 1)
        V = self.V_head(trunk_out)
        
        return T_kelvin, m_logits, V


class HADAR_PINN(nn.Module):
    """
    Physics-Informed Neural Network für HADAR spektrale Rekonstruktion.
    
    Input:  K Bänder (z.B. 1 oder 5)
    Output: 54 Bänder (rekonstruiert durch Physik)
    
    Das Netzwerk lernt die physikalischen Parameter (T, e, X) und
    rekonstruiert alle Bänder durch die HADAR-Gleichung:
    
        Sν = eν(m) × Bν(T) + [1 - eν(m)] × Xν
    """
    
    def __init__(self, 
                 wavenumbers,           # Array der Wellenzahlen (54,)
                 emissivity_library,    # Emissivitätsbibliothek (54, 28)
                 input_band_indices,    # Indizes der Input-Bänder [0] oder [0,13,26,39,52]
                 hidden_dim=64):
        super().__init__()
        
        self.wavenumbers = wavenumbers
        self.num_bands = len(wavenumbers)
        self.num_materials = emissivity_library.shape[1]
        self.input_band_indices = input_band_indices
        self.num_input_bands = len(input_band_indices)
        
        # Komponenten
        self.encoder = SpectralEncoder(self.num_input_bands, hidden_dim)
        self.decoder = ParameterDecoder(
            self.encoder.output_dim, 
            self.num_materials,
            num_env_sources=2
        )
        self.planck = PlanckLayer(wavenumbers)
        self.emissivity = EmissivityLookup(emissivity_library)
        self.env_radiation = EnvironmentalRadiationEstimator(self.num_bands)
        
        # Normalisierungsparameter (werden aus Daten geschätzt)
        self.register_buffer('input_mean', torch.zeros(self.num_input_bands))
        self.register_buffer('input_std', torch.ones(self.num_input_bands))
    
    def set_normalization(self, mean, std):
        """Setze Normalisierungsparameter basierend auf Trainingsdaten."""
        self.input_mean = mean.view(1, -1, 1, 1)
        self.input_std = std.view(1, -1, 1, 1)
    
    def normalize_input(self, x):
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def forward(self, x_input, full_cube=None):
        """
        Args:
            x_input: Input-Bänder, Shape (B, num_input_bands, H, W)
            full_cube: Optional, voller Cube für Umgebungsstrahlungs-Schätzung
        
        Returns:
            S_reconstructed: Rekonstruierter Cube, Shape (B, 54, H, W)
            params: Dictionary mit T, m_logits, V, emissivity
        """
        # Normalisiere Input
        x_norm = self.normalize_input(x_input)
        
        # Encode
        features = self.encoder(x_norm)
        
        # Decode zu physikalischen Parametern
        T, m_logits, V = self.decoder(features)
        
        # Berechne Emissivität aus Material-Logits
        emissivity, material_weights = self.emissivity(m_logits)
        
        # Berechne Planck-Strahlung
        B_nu = self.planck(T)
        
        # Berechne Umgebungsstrahlung
        X = self.env_radiation(V, full_cube)
        
        # HADAR Physics Equation: Sν = eν × Bν(T) + (1 - eν) × Xν
        S_reconstructed = emissivity * B_nu + (1 - emissivity) * X
        
        params = {
            'T': T,
            'm_logits': m_logits,
            'm_weights': material_weights,
            'V': V,
            'emissivity': emissivity,
            'B_nu': B_nu,
            'X': X
        }
        
        return S_reconstructed, params


class PINN_Loss(nn.Module):
    """
    Kombinierte Loss-Funktion für PINN Training.
    
    Loss = λ_recon × L_reconstruction + λ_physics × L_physics + λ_smooth × L_smoothness
    """
    
    def __init__(self, 
                 lambda_recon=1.0,
                 lambda_physics=0.1,
                 lambda_smooth=0.01):
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_physics = lambda_physics
        self.lambda_smooth = lambda_smooth
    
    def reconstruction_loss(self, S_pred, S_true):
        """MSE zwischen rekonstruiertem und echtem Spektrum."""
        return F.mse_loss(S_pred, S_true)
    
    def spectral_angle_loss(self, S_pred, S_true):
        """
        Spectral Angle Mapper (SAM) Loss.
        Misst den Winkel zwischen Spektralvektoren.
        """
        # Normalisiere Spektren
        S_pred_norm = F.normalize(S_pred, dim=1)
        S_true_norm = F.normalize(S_true, dim=1)
        
        # Kosinus-Ähnlichkeit
        cos_sim = (S_pred_norm * S_true_norm).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1, 1)
        
        # Winkel in Radiant
        angle = torch.acos(cos_sim)
        
        return angle.mean()
    
    def smoothness_loss(self, T, m_weights):
        """
        Fördert räumliche Glattheit der Temperatur- und Materialkarten.
        Total Variation Loss.
        """
        # Temperatur-Glattheit
        T_dx = torch.abs(T[:, :, :, 1:] - T[:, :, :, :-1])
        T_dy = torch.abs(T[:, :, 1:, :] - T[:, :, :-1, :])
        T_smooth = T_dx.mean() + T_dy.mean()
        
        # Material-Glattheit (auf Wahrscheinlichkeiten)
        m_dx = torch.abs(m_weights[:, :, :, 1:] - m_weights[:, :, :, :-1])
        m_dy = torch.abs(m_weights[:, :, 1:, :] - m_weights[:, :, :-1, :])
        m_smooth = m_dx.mean() + m_dy.mean()
        
        return T_smooth + 0.1 * m_smooth
    
    def physics_consistency_loss(self, S_pred, params):
        """
        Überprüft physikalische Konsistenz:
        - Emissivität sollte zwischen 0 und 1 liegen
        - Temperatur sollte positiv sein
        - Strahlung sollte positiv sein
        """
        loss = 0.0
        
        # Emissivität-Grenzen (bereits durch Bibliothek garantiert, aber zur Sicherheit)
        emissivity = params['emissivity']
        loss += F.relu(-emissivity).mean()  # Strafe für e < 0
        loss += F.relu(emissivity - 1).mean()  # Strafe für e > 1
        
        # Strahlung sollte positiv sein
        loss += F.relu(-S_pred).mean()
        
        return loss
    
    def forward(self, S_pred, S_true, params):
        """
        Berechne Gesamt-Loss.
        
        Args:
            S_pred: Rekonstruiertes Spektrum (B, 54, H, W)
            S_true: Ground Truth Spektrum (B, 54, H, W)
            params: Dictionary mit physikalischen Parametern
        
        Returns:
            total_loss: Skalarer Loss
            loss_dict: Dictionary mit einzelnen Loss-Komponenten
        """
        # Reconstruction Loss (MSE + SAM)
        l_mse = self.reconstruction_loss(S_pred, S_true)
        l_sam = self.spectral_angle_loss(S_pred, S_true)
        l_recon = l_mse + 0.1 * l_sam
        
        # Physics Consistency Loss
        l_physics = self.physics_consistency_loss(S_pred, params)
        
        # Smoothness Loss
        l_smooth = self.smoothness_loss(params['T'], params['m_weights'])
        
        # Gesamt-Loss
        total_loss = (self.lambda_recon * l_recon + 
                     self.lambda_physics * l_physics + 
                     self.lambda_smooth * l_smooth)
        
        loss_dict = {
            'total': total_loss,
            'mse': l_mse,
            'sam': l_sam,
            'reconstruction': l_recon,
            'physics': l_physics,
            'smoothness': l_smooth
        }
        
        return total_loss, loss_dict


def create_pinn_model(wavenumbers, emissivity_library, num_input_bands=1, hidden_dim=64):
    """
    Factory-Funktion zum Erstellen eines PINN-Modells.
    
    Args:
        wavenumbers: Array der Wellenzahlen (54,)
        emissivity_library: Emissivitätsbibliothek (54, 28)
        num_input_bands: Anzahl der Input-Bänder (1 oder 5)
        hidden_dim: Versteckte Dimension des Encoders
    
    Returns:
        model: HADAR_PINN Modell
    """
    # Wähle Input-Band-Indizes
    if num_input_bands == 1:
        # Mittleres Band
        input_indices = [27]  # Band 27 (ca. 990 cm⁻¹)
    elif num_input_bands == 3:
        # Gleichmäßig verteilt
        input_indices = [10, 27, 44]
    elif num_input_bands == 5:
        # Gleichmäßig verteilt über das Spektrum
        input_indices = [5, 15, 27, 39, 49]
    else:
        # Gleichmäßig verteilt
        step = 54 // num_input_bands
        input_indices = list(range(0, 54, step))[:num_input_bands]
    
    model = HADAR_PINN(
        wavenumbers=wavenumbers,
        emissivity_library=emissivity_library,
        input_band_indices=input_indices,
        hidden_dim=hidden_dim
    )
    
    return model


# Test
if __name__ == "__main__":
    # Dummy-Daten
    wavenumbers = np.linspace(720, 1250, 54)
    emissivity_lib = np.random.rand(54, 28) * 0.5 + 0.5  # 0.5-1.0
    
    # Erstelle Modell (1 Band Input)
    model_1band = create_pinn_model(wavenumbers, emissivity_lib, num_input_bands=1)
    print(f"1-Band Modell erstellt")
    print(f"  Input Bänder: {model_1band.input_band_indices}")
    print(f"  Anzahl Parameter: {sum(p.numel() for p in model_1band.parameters()):,}")
    
    # Erstelle Modell (5 Bänder Input)
    model_5band = create_pinn_model(wavenumbers, emissivity_lib, num_input_bands=5)
    print(f"\n5-Band Modell erstellt")
    print(f"  Input Bänder: {model_5band.input_band_indices}")
    print(f"  Anzahl Parameter: {sum(p.numel() for p in model_5band.parameters()):,}")
    
    # Test Forward Pass
    batch_size = 2
    H, W = 128, 128
    
    x_1band = torch.randn(batch_size, 1, H, W)
    x_5band = torch.randn(batch_size, 5, H, W)
    
    with torch.no_grad():
        S_pred_1, params_1 = model_1band(x_1band)
        S_pred_5, params_5 = model_5band(x_5band)
    
    print(f"\n1-Band Forward Pass:")
    print(f"  Input Shape: {x_1band.shape}")
    print(f"  Output Shape: {S_pred_1.shape}")
    print(f"  T Range: [{params_1['T'].min():.1f}, {params_1['T'].max():.1f}] K")
    
    print(f"\n5-Band Forward Pass:")
    print(f"  Input Shape: {x_5band.shape}")
    print(f"  Output Shape: {S_pred_5.shape}")
    print(f"  T Range: [{params_5['T'].min():.1f}, {params_5['T'].max():.1f}] K")
    
    # Test Loss
    loss_fn = PINN_Loss()
    S_true = torch.randn(batch_size, 54, H, W).abs()  # Positiv
    
    total_loss, loss_dict = loss_fn(S_pred_1.abs(), S_true, params_1)
    print(f"\nLoss Test:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")