"""
HADAR PINN - Model Module

Physics-Informed Neural Network für spektrale Rekonstruktion.

Input:  1 Band (z.B. Band 27, 990 cm⁻¹)
Output: 54 Bänder (rekonstruiert durch Physik)

Das Netzwerk lernt pro Pixel:
- T: Objekttemperatur
- T_air: Lufttemperatur  
- ε(ν): Emissivität (54 Werte, glatt)
- τ(ν): Atmosphärische Transmittanz (54 Werte)
- X(ν): Umgebungsstrahlung (54 Werte)

Physics Layer rekonstruiert:
    S(ν) = τ(ν) × [ε(ν) × B(ν,T) + (1-ε(ν)) × X(ν)] + (1-τ(ν)) × B(ν,T_air)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from physics import PlanckFunction, WAVENUMBERS_CM, NUM_BANDS


# =============================================================================
# Encoder: Extrahiert Features aus Input-Band
# =============================================================================

class SpatialEncoder(nn.Module):
    """
    CNN Encoder der aus dem Input-Band räumliche Features extrahiert.
    
    Verwendet mehrere Convolutional Layers mit Skip Connections
    um sowohl lokale als auch größere räumliche Kontexte zu erfassen.
    """
    
    def __init__(self, in_channels=1, base_channels=64, out_channels=256):
        super().__init__()
        
        self.out_channels = out_channels
        
        # Block 1: Input -> base_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Block 2: base_channels -> base_channels * 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Block 3: base_channels * 2 -> base_channels * 4
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Final: Kombiniere zu out_channels
        self.final = nn.Sequential(
            nn.Conv2d(base_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )
    
    def forward(self, x):
        """
        Args:
            x: Input, Shape (B, 1, H, W)
        
        Returns:
            features: Shape (B, out_channels, H, W)
        """
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        out = self.final(x3)
        
        return out


# =============================================================================
# Spectral Decoder: Latent -> 54 spektrale Werte
# =============================================================================

class SpectralDecoder(nn.Module):
    """
    Dekodiert latente Repräsentation zu 54 spektralen Werten.
    
    Die Kompression von latent_dim -> 54 erzwingt Glattheit,
    da das Netzwerk lernen muss, glatte Spektren zu erzeugen.
    """
    
    def __init__(self, latent_dim=16, num_bands=54, hidden_dim=64):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.num_bands = num_bands
        
        # MLP: latent_dim -> hidden -> num_bands
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, num_bands),
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent, Shape (B, latent_dim, H, W)
        
        Returns:
            spectrum: Shape (B, num_bands, H, W)
        """
        B, C, H, W = z.shape
        
        # Reshape für MLP: (B, C, H, W) -> (B*H*W, C)
        z_flat = z.permute(0, 2, 3, 1).reshape(B * H * W, C)
        
        # Decode
        spectrum_flat = self.decoder(z_flat)
        
        # Reshape zurück: (B*H*W, num_bands) -> (B, num_bands, H, W)
        spectrum = spectrum_flat.reshape(B, H, W, self.num_bands).permute(0, 3, 1, 2)
        
        return spectrum


# =============================================================================
# Parameter Heads: Features -> Physikalische Parameter
# =============================================================================

class ParameterHeads(nn.Module):
    """
    Separate Heads für jeden physikalischen Parameter.
    
    Aus den Encoder-Features werden geschätzt:
    - T: Objekttemperatur (1 Wert pro Pixel)
    - T_air: Lufttemperatur (1 Wert pro Pixel)
    - ε_latent: Latente Emissivität (latent_dim Werte)
    - τ_latent: Latente Transmittanz (latent_dim Werte)
    - X_latent: Latente Umgebungsstrahlung (latent_dim Werte)
    """
    
    def __init__(self, in_channels=256, latent_dim=16):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Gemeinsamer Trunk
        self.trunk = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        trunk_out = in_channels // 2
        
        # T Head: Objekttemperatur
        self.T_head = nn.Sequential(
            nn.Conv2d(trunk_out, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        # T_air Head: Lufttemperatur
        self.T_air_head = nn.Sequential(
            nn.Conv2d(trunk_out, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )
        
        # ε Head: Emissivität (latent)
        self.eps_head = nn.Sequential(
            nn.Conv2d(trunk_out, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=1),
        )
        
        # τ Head: Transmittanz (latent)
        self.tau_head = nn.Sequential(
            nn.Conv2d(trunk_out, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=1),
        )
        
        # X Head: Umgebungsstrahlung (latent)
        self.X_head = nn.Sequential(
            nn.Conv2d(trunk_out, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, latent_dim, kernel_size=1),
        )
        
        # Initialisierung
        self._init_weights()
    
    def _init_weights(self):
        """Initialisiere Gewichte für sinnvolle Startwerte."""
        # T Head: Start bei ~290 K (17°C)
        nn.init.zeros_(self.T_head[-1].weight)
        nn.init.constant_(self.T_head[-1].bias, 0.4)  # sigmoid(0.4)*100+250 ≈ 290
        
        # T_air Head: Start bei ~285 K (12°C)
        nn.init.zeros_(self.T_air_head[-1].weight)
        nn.init.constant_(self.T_air_head[-1].bias, 0.3)  # sigmoid(0.3)*70+250 ≈ 277
    
    def forward(self, features):
        """
        Args:
            features: Encoder output, Shape (B, in_channels, H, W)
        
        Returns:
            Dict mit T, T_air, eps_latent, tau_latent, X_latent
        """
        trunk = self.trunk(features)
        
        # Temperaturen: Sigmoid skaliert auf physikalischen Bereich
        T_raw = self.T_head(trunk)
        T = torch.sigmoid(T_raw) * 100 + 250  # [250, 350] K
        
        T_air_raw = self.T_air_head(trunk)
        T_air = torch.sigmoid(T_air_raw) * 70 + 250  # [250, 320] K
        
        # Latente Repräsentationen
        eps_latent = self.eps_head(trunk)
        tau_latent = self.tau_head(trunk)
        X_latent = self.X_head(trunk)
        
        return {
            'T': T,
            'T_air': T_air,
            'eps_latent': eps_latent,
            'tau_latent': tau_latent,
            'X_latent': X_latent,
        }


# =============================================================================
# Physics Layer: Rekonstruiert Spektrum aus Parametern
# =============================================================================

class PhysicsLayer(nn.Module):
    """
    Physik-Layer der die HADAR-Gleichung implementiert.
    
    S(ν) = τ(ν) × [ε(ν) × B(ν,T) + (1-ε(ν)) × X(ν)] + (1-τ(ν)) × B(ν,T_air)
    
    Dieser Layer ist nicht trainierbar - er wendet nur die Physik an.
    """
    
    def __init__(self, wavenumbers_cm=None):
        super().__init__()
        
        if wavenumbers_cm is None:
            wavenumbers_cm = WAVENUMBERS_CM
        
        self.planck = PlanckFunction(wavenumbers_cm)
        self.num_bands = len(wavenumbers_cm)
    
    def forward(self, T, T_air, epsilon, tau, X):
        """
        Berechne Strahlung mit vollständiger Atmosphären-Gleichung.
        
        Args:
            T: Objekttemperatur, Shape (B, 1, H, W), [K]
            T_air: Lufttemperatur, Shape (B, 1, H, W), [K]
            epsilon: Emissivität, Shape (B, 54, H, W), [0,1]
            tau: Transmittanz, Shape (B, 54, H, W), [0,1]
            X: Umgebungsstrahlung, Shape (B, 54, H, W), [≥0]
        
        Returns:
            S: Beobachtete Strahlung, Shape (B, 54, H, W)
        """
        # Planck-Strahlung für Objekt und Luft
        B_T = self.planck(T)          # (B, 54, H, W)
        B_T_air = self.planck(T_air)  # (B, 54, H, W)
        
        # Objektemission: ε × B(T)
        object_emission = epsilon * B_T
        
        # Reflektierte Umgebungsstrahlung: (1-ε) × X
        reflected = (1 - epsilon) * X
        
        # Gesamte Objektstrahlung (vor Atmosphäre)
        object_total = object_emission + reflected
        
        # Durch Atmosphäre: τ × object_total
        through_atmosphere = tau * object_total
        
        # Atmosphärische Emission: (1-τ) × B(T_air)
        atmosphere_emission = (1 - tau) * B_T_air
        
        # Gesamte beobachtete Strahlung
        S = through_atmosphere + atmosphere_emission
        
        return S


# =============================================================================
# Vollständiges PINN Modell
# =============================================================================

class HADAR_PINN_v2(nn.Module):
    """
    Physics-Informed Neural Network für HADAR spektrale Rekonstruktion.
    
    Version 2: Mit Atmosphäre und gelernten Parametern.
    
    Input:  1 Band
    Output: 54 Bänder (durch Physik rekonstruiert)
    
    Zusätzlich werden geschätzt:
    - T: Temperaturkarte
    - T_air: Lufttemperaturkarte
    - ε(ν): Emissivitätsspektren
    - τ(ν): Transmittanzspektren
    - X(ν): Umgebungsstrahlungsspektren
    """
    
    def __init__(self, 
                 input_band_idx=27,
                 latent_dim=16,
                 encoder_channels=256,
                 base_channels=64):
        super().__init__()
        
        self.input_band_idx = input_band_idx
        self.latent_dim = latent_dim
        self.num_bands = NUM_BANDS  # 54
        
        # Encoder
        self.encoder = SpatialEncoder(
            in_channels=1,
            base_channels=base_channels,
            out_channels=encoder_channels
        )
        
        # Parameter Heads
        self.param_heads = ParameterHeads(
            in_channels=encoder_channels,
            latent_dim=latent_dim
        )
        
        # Spektrale Decoder
        self.eps_decoder = SpectralDecoder(latent_dim, self.num_bands)
        self.tau_decoder = SpectralDecoder(latent_dim, self.num_bands)
        self.X_decoder = SpectralDecoder(latent_dim, self.num_bands)
        
        # Physics Layer
        self.physics = PhysicsLayer()
        
        # Statistiken für Normalisierung (werden von außen gesetzt)
        self.register_buffer('input_mean', torch.tensor(0.0))
        self.register_buffer('input_std', torch.tensor(1.0))
    
    def set_normalization(self, mean, std):
        """Setze Normalisierungsparameter."""
        self.input_mean = torch.tensor(mean, dtype=torch.float32)
        self.input_std = torch.tensor(std, dtype=torch.float32)
    
    def forward(self, x):
        """
        Forward Pass.
        
        Args:
            x: Input Band, Shape (B, 1, H, W)
        
        Returns:
            S_pred: Rekonstruiertes Spektrum, Shape (B, 54, H, W)
            params: Dict mit allen geschätzten Parametern
        """
        # Normalisiere Input
        x_norm = (x - self.input_mean) / (self.input_std + 1e-8)
        
        # Encode
        features = self.encoder(x_norm)
        
        # Parameter Heads
        params_latent = self.param_heads(features)
        
        # Dekodiere spektrale Parameter
        eps_raw = self.eps_decoder(params_latent['eps_latent'])
        tau_raw = self.tau_decoder(params_latent['tau_latent'])
        X_raw = self.X_decoder(params_latent['X_latent'])
        
        # Aktivierungen für physikalische Constraints
        epsilon = torch.sigmoid(eps_raw)  # [0, 1]
        tau = torch.sigmoid(tau_raw)       # [0, 1]
        X = F.softplus(X_raw)              # [0, ∞)
        
        # Physics Layer
        S_pred = self.physics(
            T=params_latent['T'],
            T_air=params_latent['T_air'],
            epsilon=epsilon,
            tau=tau,
            X=X
        )
        
        # Sammle alle Parameter
        params = {
            'T': params_latent['T'],
            'T_air': params_latent['T_air'],
            'epsilon': epsilon,
            'tau': tau,
            'X': X,
            'eps_latent': params_latent['eps_latent'],
            'tau_latent': params_latent['tau_latent'],
            'X_latent': params_latent['X_latent'],
        }
        
        return S_pred, params
    
    def get_num_params(self):
        """Zähle trainierbare Parameter."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Loss-Funktionen
# =============================================================================

class PINN_Loss_v2(nn.Module):
    """
    Loss-Funktion für HADAR PINN v2.

    L_total = L_reconstruction + λ_sam × L_sam + λ_smooth × L_smoothness + λ_physics × L_physics

    - L_reconstruction: MSE zwischen rekonstruiertem und echtem Spektrum
    - L_sam: Spectral Angle Mapper Loss
    - L_smoothness: Fördert glatte ε und τ Spektren
    - L_physics: Bestraft unphysikalische Werte
    """

    def __init__(self,
                 lambda_recon=1.0,
                 lambda_sam=0.1,
                 lambda_smooth=0.01,
                 lambda_physics=0.001,
                 lambda_temp=0.001,
                 lambda_consistency=0.01):
        super().__init__()

        self.lambda_recon = lambda_recon
        self.lambda_sam = lambda_sam
        self.lambda_smooth = lambda_smooth
        self.lambda_physics = lambda_physics
        self.lambda_temp = lambda_temp
        self.lambda_consistency = lambda_consistency
    
    def smoothness_loss(self, spectrum):
        """
        Berechne Glattheitsverlust (Total Variation in spektraler Dimension).
        
        Args:
            spectrum: Shape (B, num_bands, H, W)
        
        Returns:
            Skalarer Loss
        """
        # Differenz zwischen benachbarten Bändern
        diff = spectrum[:, 1:, :, :] - spectrum[:, :-1, :, :]
        return torch.mean(diff ** 2)
    
    def physics_loss(self, params):
        """
        Bestraft unphysikalische Werte.
        
        - T sollte nahe T_air sein (natürliche Szenen)
        - ε sollte typischerweise hoch sein (> 0.5 für natürliche Materialien)
        - τ sollte nahe 1 sein für kurze Distanzen
        """
        loss = 0.0
        
        # Temperatur-Regularisierung: T und T_air sollten nicht zu unterschiedlich sein
        T_diff = torch.abs(params['T'] - params['T_air'])
        loss += torch.mean(F.relu(T_diff - 30))  # Strafe wenn |T - T_air| > 30 K
        
        # Emissivität: Fördere typische Werte (0.8-1.0 für natürliche Materialien)
        eps_mean = params['epsilon'].mean(dim=1, keepdim=True)
        loss += 0.1 * torch.mean(F.relu(0.5 - eps_mean))  # Strafe wenn ε < 0.5
        
        # Transmittanz: Für kurze Distanzen sollte τ nahe 1 sein
        tau_mean = params['tau'].mean(dim=1, keepdim=True)
        loss += 0.1 * torch.mean(F.relu(0.7 - tau_mean))  # Strafe wenn τ < 0.7
        
        return loss
    
    def consistency_loss(self, S_pred, S_input, input_band_idx):
        """
        Das rekonstruierte Input-Band sollte dem echten Input entsprechen.
        
        Args:
            S_pred: Rekonstruiertes Spektrum (B, 54, H, W)
            S_input: Echtes Input-Band (B, 1, H, W)
            input_band_idx: Index des Input-Bands
        """
        S_pred_input = S_pred[:, input_band_idx:input_band_idx+1, :, :]
        return F.mse_loss(S_pred_input, S_input)
    
    def sam_loss(self, S_pred, S_true):
        """
        Spectral Angle Mapper Loss.

        Misst den Winkel zwischen Spektren - unabhängig von der Amplitude.
        """
        # Normalisiere entlang der spektralen Dimension
        S_pred_norm = F.normalize(S_pred, p=2, dim=1)
        S_true_norm = F.normalize(S_true, p=2, dim=1)

        # Kosinus-Ähnlichkeit
        cos_sim = (S_pred_norm * S_true_norm).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        # Winkel in Radiant
        sam = torch.acos(cos_sim)

        return sam.mean()

    def forward(self, S_pred, S_true, params, S_input=None, input_band_idx=27, compute_residual=False):
        """
        Berechne Gesamtverlust.

        Args:
            S_pred: Rekonstruiertes Spektrum (B, 54, H, W)
            S_true: Echtes Spektrum (B, 54, H, W)
            params: Dict mit Parametern
            S_input: Optional, Input-Band für Konsistenz-Loss
            input_band_idx: Index des Input-Bands
            compute_residual: Ob Residuum berechnet werden soll

        Returns:
            total_loss: Skalarer Loss
            loss_dict: Dict mit einzelnen Komponenten
        """
        # Rekonstruktionsverlust (Hauptloss)
        loss_recon = F.mse_loss(S_pred, S_true)

        # SAM Loss
        loss_sam = self.sam_loss(S_pred, S_true)

        # Glattheitsverlust für ε und τ
        loss_smooth_eps = self.smoothness_loss(params['epsilon'])
        loss_smooth_tau = self.smoothness_loss(params['tau'])
        loss_smooth = loss_smooth_eps + loss_smooth_tau

        # Physikverlust
        loss_physics = self.physics_loss(params)

        # Temperatur-Regularisierung
        loss_temp = torch.tensor(0.0, device=S_pred.device)
        if 'T' in params:
            # Temperatur sollte in physikalischem Bereich bleiben
            T = params['T']
            loss_temp = torch.mean(F.relu(T - 350)) + torch.mean(F.relu(250 - T))

        # Konsistenzverlust (optional)
        loss_consistency = torch.tensor(0.0, device=S_pred.device)
        if S_input is not None:
            loss_consistency = self.consistency_loss(S_pred, S_input, input_band_idx)

        # Residuum berechnen
        residual_abs_mean = 0.0
        if compute_residual:
            residual = torch.abs(S_pred - S_true)
            residual_abs_mean = residual.mean().item()

        # Gesamtverlust
        total_loss = (self.lambda_recon * loss_recon +
                     self.lambda_sam * loss_sam +
                     self.lambda_smooth * loss_smooth +
                     self.lambda_physics * loss_physics +
                     self.lambda_temp * loss_temp +
                     self.lambda_consistency * loss_consistency)

        loss_dict = {
            'total': total_loss,
            'reconstruction': loss_recon,
            'sam': loss_sam,
            'smoothness': loss_smooth,
            'smoothness_eps': loss_smooth_eps,
            'smoothness_tau': loss_smooth_tau,
            'physics': loss_physics,
            'temp': loss_temp,
            'consistency': loss_consistency,
            'residual_abs_mean': residual_abs_mean,
        }

        return total_loss, loss_dict


# =============================================================================
# Evaluation Metrics
# =============================================================================

class EvaluationMetrics:
    """Sammlung von Evaluierungsmetriken."""

    @staticmethod
    def psnr(pred, target, max_val=None):
        """
        Peak Signal-to-Noise Ratio.

        Args:
            pred: Vorhersage Tensor
            target: Ground Truth Tensor
            max_val: Maximaler Wert (falls None, wird aus target berechnet)

        Returns:
            PSNR in dB
        """
        if max_val is None:
            max_val = target.max().item()

        mse = F.mse_loss(pred, target).item()
        if mse < 1e-10:
            return 100.0  # Perfekte Rekonstruktion

        psnr = 10 * np.log10(max_val**2 / mse)
        return psnr

    @staticmethod
    def sam(pred, target):
        """
        Spectral Angle Mapper (in Grad).

        Args:
            pred: Vorhersage (B, C, H, W)
            target: Ground Truth (B, C, H, W)

        Returns:
            Mittlerer SAM in Grad
        """
        pred_norm = F.normalize(pred, p=2, dim=1)
        target_norm = F.normalize(target, p=2, dim=1)

        cos_sim = (pred_norm * target_norm).sum(dim=1)
        cos_sim = torch.clamp(cos_sim, -1.0, 1.0)

        sam_rad = torch.acos(cos_sim)
        sam_deg = sam_rad * 180.0 / np.pi

        return sam_deg.mean().item()

    @staticmethod
    def rmse(pred, target):
        """Root Mean Square Error."""
        return torch.sqrt(F.mse_loss(pred, target)).item()

    @staticmethod
    def mae(pred, target):
        """Mean Absolute Error."""
        return F.l1_loss(pred, target).item()


# =============================================================================
# Factory Function
# =============================================================================

def create_pinn_model(wavenumbers, input_band_idx=27, dim_latent=16,
                      hidden_dim=64, T_min=260.0, T_max=320.0,
                      include_atmosphere=True):
    """
    Factory-Funktion zum Erstellen eines HADAR PINN Modells.

    Args:
        wavenumbers: Array mit Wellenzahlen in cm⁻¹
        input_band_idx: Index des Input-Bands
        dim_latent: Latente Dimension für spektrale Decoder
        hidden_dim: Versteckte Dimension (wird für base_channels verwendet)
        T_min: Minimale Temperatur (für Dokumentation)
        T_max: Maximale Temperatur (für Dokumentation)
        include_atmosphere: Ob Atmosphäre modelliert werden soll

    Returns:
        HADAR_PINN_v2 Modell
    """
    model = HADAR_PINN_v2(
        input_band_idx=input_band_idx,
        latent_dim=dim_latent,
        encoder_channels=256,
        base_channels=hidden_dim
    )

    # Speichere zusätzliche Infos als Attribute
    model.T_min = T_min
    model.T_max = T_max
    model.include_atmosphere = include_atmosphere

    return model


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("HADAR PINN v2 - Model Test")
    print("=" * 60)
    
    # Erstelle Modell
    model = HADAR_PINN_v2(
        input_band_idx=27,
        latent_dim=16,
        encoder_channels=256,
        base_channels=64
    )
    
    print(f"\nModell erstellt:")
    print(f"  Input Band Index: {model.input_band_idx}")
    print(f"  Latent Dimension: {model.latent_dim}")
    print(f"  Anzahl Bänder: {model.num_bands}")
    print(f"  Trainierbare Parameter: {model.get_num_params():,}")
    
    # Test Forward Pass
    print(f"\nForward Pass Test:")
    batch_size = 2
    H, W = 64, 64
    
    x = torch.rand(batch_size, 1, H, W) * 0.1 + 0.05  # Simulierte Strahlung
    
    with torch.no_grad():
        S_pred, params = model(x)
    
    print(f"  Input Shape: {x.shape}")
    print(f"  Output Shape: {S_pred.shape}")
    print(f"  T Range: [{params['T'].min():.1f}, {params['T'].max():.1f}] K")
    print(f"  T_air Range: [{params['T_air'].min():.1f}, {params['T_air'].max():.1f}] K")
    print(f"  ε Range: [{params['epsilon'].min():.3f}, {params['epsilon'].max():.3f}]")
    print(f"  τ Range: [{params['tau'].min():.3f}, {params['tau'].max():.3f}]")
    print(f"  X Range: [{params['X'].min():.2e}, {params['X'].max():.2e}]")
    
    # Test Loss
    print(f"\nLoss Test:")
    loss_fn = PINN_Loss_v2()
    
    S_true = torch.rand(batch_size, 54, H, W) * 0.1 + 0.05
    total_loss, loss_dict = loss_fn(S_pred, S_true, params, x, 27)
    
    print(f"  Total Loss: {loss_dict['total'].item():.6f}")
    print(f"  Reconstruction: {loss_dict['reconstruction'].item():.6f}")
    print(f"  Smoothness: {loss_dict['smoothness'].item():.6f}")
    print(f"  Physics: {loss_dict['physics'].item():.6f}")
    print(f"  Consistency: {loss_dict['consistency'].item():.6f}")
    
    print("\n" + "=" * 60)
    print("Model Test abgeschlossen!")
    print("=" * 60)
