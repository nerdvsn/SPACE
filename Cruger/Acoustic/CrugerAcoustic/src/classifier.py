"""
Cruger Acoustic - Classifier
Lädt das trainierte Modell und macht Vorhersagen
"""

import os
from pathlib import Path
from fastai.vision.all import load_learner, PILImage
import torch


class SoundClassifier:
    """Lädt und verwendet das trainierte UrbanSound8K Modell"""
    
    # Die 10 Klassen aus dem Training
    LABELS = [
        'air_conditioner',
        'car_horn', 
        'children_playing',
        'dog_bark',
        'drilling',
        'engine_idling',
        'gun_shot',
        'jackhammer',
        'siren',
        'street_music'
    ]
    
    # Deutsche Übersetzungen für die UI
    LABELS_DE = {
        'air_conditioner': 'Klimaanlage',
        'car_horn': 'Autohupe',
        'children_playing': 'Spielende Kinder',
        'dog_bark': 'Hundebellen',
        'drilling': 'Bohrmaschine',
        'engine_idling': 'Motor (Leerlauf)',
        'gun_shot': 'Schuss',
        'jackhammer': 'Presslufthammer',
        'siren': 'Sirene',
        'street_music': 'Straßenmusik'
    }
    
    # Emojis für die UI
    LABEL_EMOJIS = {
        'air_conditioner': '❄️',
        'car_horn': '🚗',
        'children_playing': '👧',
        'dog_bark': '🐕',
        'drilling': '🔧',
        'engine_idling': '🚙',
        'gun_shot': '💥',
        'jackhammer': '🔨',
        'siren': '🚨',
        'street_music': '🎵'
    }
    
    def __init__(self, model_path=None):
        self.model = None
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    def load_model(self, model_path=None):
        """Lädt das trainierte fastai Modell"""
        if model_path:
            self.model_path = model_path
            
        if not self.model_path:
            raise ValueError("Kein Modellpfad angegeben!")
            
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Modell nicht gefunden: {self.model_path}")
        
        print(f"Lade Modell von: {self.model_path}")
        print(f"Verwende Device: {self.device}")
        
        self.model = load_learner(self.model_path)
        
        # GPU wenn verfügbar
        if self.device == 'cuda':
            self.model.model.cuda()
            
        print("✓ Modell erfolgreich geladen!")
        return True
    
    def predict(self, spectrogram_image):
        """
        Macht eine Vorhersage für ein Spektrogramm-Bild
        
        Args:
            spectrogram_image: PIL Image (224x224)
            
        Returns:
            dict mit 'label', 'label_de', 'emoji', 'confidence', 'all_probs'
        """
        if self.model is None:
            raise RuntimeError("Modell nicht geladen! Rufe zuerst load_model() auf.")
        
        if spectrogram_image is None:
            return None
        
        # fastai erwartet PILImage
        img = PILImage.create(spectrogram_image)
        
        # Vorhersage
        pred_class, pred_idx, probs = self.model.predict(img)
        
        # Alle Wahrscheinlichkeiten als Dictionary
        all_probs = {}
        for i, label in enumerate(self.LABELS):
            all_probs[label] = float(probs[i])
        
        # Ergebnis zusammenstellen
        result = {
            'label': str(pred_class),
            'label_de': self.LABELS_DE.get(str(pred_class), str(pred_class)),
            'emoji': self.LABEL_EMOJIS.get(str(pred_class), '🔊'),
            'confidence': float(probs[pred_idx]),
            'all_probs': all_probs
        }
        
        return result
    
    def get_top_predictions(self, spectrogram_image, top_k=3):
        """Gibt die Top-K Vorhersagen zurück"""
        result = self.predict(spectrogram_image)
        if result is None:
            return None
        
        # Sortiere nach Wahrscheinlichkeit
        sorted_probs = sorted(
            result['all_probs'].items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        top_predictions = []
        for label, prob in sorted_probs[:top_k]:
            top_predictions.append({
                'label': label,
                'label_de': self.LABELS_DE.get(label, label),
                'emoji': self.LABEL_EMOJIS.get(label, '🔊'),
                'confidence': prob
            })
        
        return top_predictions


if __name__ == "__main__":
    # Test
    classifier = SoundClassifier()
    print("Sound Classifier initialisiert")
    print(f"Klassen: {classifier.LABELS}")
    print(f"Device: {classifier.device}")
