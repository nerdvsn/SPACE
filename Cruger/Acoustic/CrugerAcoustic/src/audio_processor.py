"""
Cruger Acoustic - Audio Processor
Nimmt Audio vom Mikrofon auf und erstellt Spektrogramme
"""

import numpy as np
import librosa
import sounddevice as sd
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa.display


class AudioProcessor:
    """Verarbeitet Audio-Input und erstellt Mel-Spektrogramme"""
    
    def __init__(self, duration=1.0):
        self.duration = duration
        self.target_sr = 22050  # Für librosa/Modell
        self.device_sr = None   # Wird vom Gerät geholt
        self.chunk_samples = None
        
    def get_available_devices(self):
        """Gibt Liste aller verfügbaren Audio-Eingabegeräte zurück"""
        devices = sd.query_devices()
        input_devices = []
        
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                input_devices.append({
                    'index': i,
                    'name': dev['name'],
                    'channels': dev['max_input_channels'],
                    'sample_rate': int(dev['default_samplerate'])
                })
        
        return input_devices
    
    def get_device_sample_rate(self, device_index):
        """Holt die native Sample Rate eines Geräts"""
        if device_index is None:
            return 44100  # Fallback
        
        try:
            dev = sd.query_devices(device_index)
            return int(dev['default_samplerate'])
        except:
            return 44100
    
    def record_chunk(self, device_index=None):
        """Nimmt einen Audio-Chunk auf"""
        try:
            # Hole native Sample Rate vom Gerät
            self.device_sr = self.get_device_sample_rate(device_index)
            self.chunk_samples = int(self.device_sr * self.duration)
            
            recording = sd.rec(
                frames=self.chunk_samples,
                samplerate=self.device_sr,
                channels=1,
                dtype='float32',
                device=device_index
            )
            sd.wait()
            
            audio = recording.flatten()
            
            # Resample zu 22050 Hz für das Modell (wie im Training)
            if self.device_sr != self.target_sr:
                audio = librosa.resample(
                    audio, 
                    orig_sr=self.device_sr, 
                    target_sr=self.target_sr
                )
            
            return audio
            
        except Exception as e:
            print(f"Aufnahme-Fehler: {e}")
            return None
    
    def audio_to_spectrogram(self, audio_data):
        """Konvertiert Audio-Daten zu Mel-Spektrogramm (wie im Training)"""
        if audio_data is None or len(audio_data) == 0:
            return None
        
        # Mel-Spektrogramm erstellen (gleiche Parameter wie Training!)
        S = librosa.feature.melspectrogram(
            y=audio_data, 
            sr=self.target_sr
        )
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # Als Bild speichern (gleiche Größe wie Training: 0.72 x 0.72 inch @ 400 dpi)
        fig = plt.figure(figsize=[0.72, 0.72])
        ax = fig.add_subplot(111)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        ax.set_frame_on(False)
        
        librosa.display.specshow(S_db)
        
        # In PIL Image konvertieren
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=400, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        img = Image.open(buf).convert('RGB')
        return img
    
    def get_spectrogram_for_prediction(self, audio_data):
        """Erstellt Spektrogramm und bereitet es für das Modell vor"""
        img = self.audio_to_spectrogram(audio_data)
        if img is None:
            return None
        
        # Resize auf 224x224 (wie im Training)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)
        return img


if __name__ == "__main__":
    # Test
    processor = AudioProcessor()
    print("Verfügbare Mikrofone:")
    for dev in processor.get_available_devices():
        print(f"  [{dev['index']}] {dev['name']} @ {dev['sample_rate']} Hz")