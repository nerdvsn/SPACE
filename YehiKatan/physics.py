"""
HADAR PINN - Physics Module

Enthält alle physikalischen Funktionen und Konstanten:
- Planck-Funktion B(ν, T)
- Physikalische Konstanten

Wellenzahlen: 720 - 1250 cm⁻¹ (54 Bänder, Schritt 10)
"""

import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# Physikalische Konstanten
# =============================================================================

H_PLANCK = 6.62607015e-34      # Planck-Konstante [J·s]
C_LIGHT = 2.99792458e8         # Lichtgeschwindigkeit [m/s]
K_BOLTZMANN = 1.380649e-23     # Boltzmann-Konstante [J/K]

# Abgeleitete Konstante für Planck-Formel
C1 = 2 * H_PLANCK * C_LIGHT**2  # Erste Strahlungskonstante
C2 = H_PLANCK * C_LIGHT / K_BOLTZMANN  # Zweite Strahlungskonstante [m·K]

# =============================================================================
# Wellenzahlen
# =============================================================================

# HADAR Wellenzahlen: 720 bis 1250 cm⁻¹ in 10er Schritten
WAVENUMBERS_CM = np.arange(720, 1260, 10, dtype=np.float32)  # [cm⁻¹]
NUM_BANDS = len(WAVENUMBERS_CM)  # 54 Bänder

# Umrechnung zu SI-Einheiten
WAVENUMBERS_M = WAVENUMBERS_CM * 100  # [m⁻¹]
WAVELENGTHS_M = 1.0 / WAVENUMBERS_M   # [m]
WAVELENGTHS_UM = WAVELENGTHS_M * 1e6  # [μm]


# =============================================================================
# Planck-Funktion
# =============================================================================

class PlanckFunction(nn.Module):
    """
    Berechnet die Planck-Schwarzkörperstrahlung B(ν, T).
    
    Formel (in Wellenzahl-Form):
        B(ν, T) = 2hc²ν³ / (exp(hcν/kT) - 1)
    
    Wobei ν in [m⁻¹] und T in [K].
    
    Output ist in [W/(m² sr m⁻¹)] - Spektrale Strahldichte pro Wellenzahl.
    """
    
    def __init__(self, wavenumbers_cm=None):
        super().__init__()
        
        if wavenumbers_cm is None:
            wavenumbers_cm = WAVENUMBERS_CM
        
        # Wellenzahlen in m⁻¹
        wavenumbers_m = torch.tensor(wavenumbers_cm * 100, dtype=torch.float32)
        self.register_buffer('nu', wavenumbers_m)  # [m⁻¹]
        self.num_bands = len(wavenumbers_cm)
        
        # Vorberechnete Konstanten für Effizienz
        # C1 = 2hc² und C2 = hc/k
        self.register_buffer('c1', torch.tensor(C1, dtype=torch.float32))
        self.register_buffer('c2', torch.tensor(C2, dtype=torch.float32))
    
    def forward(self, T):
        """
        Berechne Planck-Strahlung für gegebene Temperatur.
        
        Args:
            T: Temperatur in Kelvin, Shape (B, 1, H, W)
        
        Returns:
            B_nu: Spektrale Strahldichte, Shape (B, num_bands, H, W)
        """
        # nu: (num_bands,) -> (1, num_bands, 1, 1)
        nu = self.nu.view(1, -1, 1, 1)
        
        # T: (B, 1, H, W)
        # Verhindere Division durch 0 und numerische Probleme
        T_safe = torch.clamp(T, min=200.0, max=400.0)
        
        # Exponent: hcν / kT = c2 * ν / T
        exponent = self.c2 * nu / T_safe
        
        # Numerische Stabilität: Begrenze Exponent
        exponent = torch.clamp(exponent, min=1e-10, max=50.0)
        
        # Planck-Formel: B = 2hc²ν³ / (exp(hcν/kT) - 1)
        # = c1 * ν³ / (exp(c2*ν/T) - 1)
        numerator = self.c1 * (nu ** 3)
        denominator = torch.exp(exponent) - 1.0
        
        # Verhindere Division durch 0
        denominator = torch.clamp(denominator, min=1e-10)
        
        B_nu = numerator / denominator
        
        return B_nu


# =============================================================================
# Hilfsfunktionen
# =============================================================================

def temperature_to_radiance(T, wavenumbers_cm=None):
    """
    Konvertiere Temperatur zu Schwarzkörper-Strahlung.
    
    Args:
        T: Temperatur in Kelvin (kann Tensor oder float sein)
        wavenumbers_cm: Wellenzahlen in cm⁻¹
    
    Returns:
        B: Spektrale Strahldichte für jede Wellenzahl
    """
    if wavenumbers_cm is None:
        wavenumbers_cm = WAVENUMBERS_CM
    
    if not isinstance(T, torch.Tensor):
        T = torch.tensor(T, dtype=torch.float32)
    
    if T.dim() == 0:
        T = T.view(1, 1, 1, 1)
    elif T.dim() == 1:
        T = T.view(-1, 1, 1, 1)
    
    planck = PlanckFunction(wavenumbers_cm)
    return planck(T)


def normalize_spectrum(S, method='minmax'):
    """
    Normalisiere Spektrum.
    
    Args:
        S: Spektrum, Shape (..., num_bands, H, W) oder (..., num_bands)
        method: 'minmax', 'zscore', oder 'sum'
    
    Returns:
        S_norm: Normalisiertes Spektrum
    """
    if method == 'minmax':
        S_min = S.min(dim=-3, keepdim=True)[0] if S.dim() >= 3 else S.min()
        S_max = S.max(dim=-3, keepdim=True)[0] if S.dim() >= 3 else S.max()
        return (S - S_min) / (S_max - S_min + 1e-8)
    
    elif method == 'zscore':
        S_mean = S.mean(dim=-3, keepdim=True) if S.dim() >= 3 else S.mean()
        S_std = S.std(dim=-3, keepdim=True) if S.dim() >= 3 else S.std()
        return (S - S_mean) / (S_std + 1e-8)
    
    elif method == 'sum':
        S_sum = S.sum(dim=-3, keepdim=True) if S.dim() >= 3 else S.sum()
        return S / (S_sum + 1e-8)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HADAR PINN - Physics Module Test")
    print("=" * 60)
    
    # Wellenzahlen Info
    print(f"\nWellenzahlen:")
    print(f"  Bereich: {WAVENUMBERS_CM[0]} - {WAVENUMBERS_CM[-1]} cm⁻¹")
    print(f"  Anzahl Bänder: {NUM_BANDS}")
    print(f"  Wellenlängen: {WAVELENGTHS_UM[0]:.2f} - {WAVELENGTHS_UM[-1]:.2f} μm")
    
    # Planck-Funktion Test
    print(f"\nPlanck-Funktion Test:")
    planck = PlanckFunction()
    
    # Test mit verschiedenen Temperaturen
    temps = [250, 273, 300, 350]  # Kelvin
    
    for T in temps:
        T_tensor = torch.tensor([[[[T]]]], dtype=torch.float32)
        B = planck(T_tensor)
        print(f"  T = {T} K:")
        print(f"    B_min = {B.min().item():.2e} W/(m² sr m⁻¹)")
        print(f"    B_max = {B.max().item():.2e} W/(m² sr m⁻¹)")
        print(f"    B_mean = {B.mean().item():.2e} W/(m² sr m⁻¹)")
    
    # Batch Test
    print(f"\nBatch Test:")
    T_batch = torch.tensor([280, 300, 320], dtype=torch.float32).view(3, 1, 1, 1)
    B_batch = planck(T_batch)
    print(f"  Input Shape: {T_batch.shape}")
    print(f"  Output Shape: {B_batch.shape}")
    
    # Räumlicher Test
    print(f"\nRäumlicher Test:")
    T_spatial = torch.rand(2, 1, 64, 64) * 50 + 275  # 275-325 K
    B_spatial = planck(T_spatial)
    print(f"  Input Shape: {T_spatial.shape}")
    print(f"  Output Shape: {B_spatial.shape}")
    
    print("\n" + "=" * 60)
    print("Physics Module Test abgeschlossen!")
    print("=" * 60)
