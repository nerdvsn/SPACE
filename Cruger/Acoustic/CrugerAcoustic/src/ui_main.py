"""
Cruger Acoustic - Jarvis/Defense Edition
Futuristic UI for Real-time Sound Classification
"""

import sys
import numpy as np
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QProgressBar, QFileDialog,
    QFrame, QGroupBox, QMessageBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QPalette, QColor, QPainter, QPen

from audio_processor import AudioProcessor
from classifier import SoundClassifier

# --- FUTURISTIC STYLE CONFIG ---
THEME = {
    "bg": "#070b10",         # Tiefes Weltraum-Blau-Schwarz
    "panel": "#0d141f",      # Panel-Hintergrund
    "accent": "#00f0ff",     # Jarvis-Cyan
    "accent_low": "#004e52", # Abgedunkeltes Cyan
    "text": "#e0f7fa",       # Helles Text-Blau
    "danger": "#ff3b3b",     # Alarm-Rot
    "border": "#1a2a3a"      # Dezente Rahmen
}

STYLESHEET = f"""
    QMainWindow {{
        background-color: {THEME['bg']};
    }}
    
    QGroupBox {{
        color: {THEME['accent']};
        font-family: 'Segoe UI Semibold', 'Consolas';
        font-size: 13px;
        border: 1px solid {THEME['border']};
        margin-top: 15px;
        border-radius: 2px;
        background-color: {THEME['panel']};
    }}
    
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
    }}

    QLabel {{
        color: {THEME['text']};
        font-family: 'Consolas', 'Monospace';
    }}

    QPushButton {{
        background-color: transparent;
        color: {THEME['accent']};
        border: 1px solid {THEME['accent']};
        border-radius: 0px;
        padding: 8px;
        font-family: 'Segoe UI Semibold';
        text-transform: uppercase;
        letter-spacing: 2px;
    }}

    QPushButton:hover {{
        background-color: {THEME['accent_low']};
        border: 1px solid {THEME['accent']};
    }}
    
    QPushButton:disabled {{
        color: #444;
        border: 1px solid #444;
    }}

    QProgressBar {{
        border: 1px solid {THEME['accent_low']};
        border-radius: 0px;
        background-color: #05080c;
        text-align: right;
        margin-right: 40px;
    }}

    QProgressBar::chunk {{
        background-color: {THEME['accent']};
        width: 4px;
        margin: 0.5px;
    }}

    QComboBox {{
        background-color: {THEME['bg']};
        color: {THEME['accent']};
        border: 1px solid {THEME['border']};
        padding: 5px;
    }}
"""

class AudioWorker(QThread):
    result_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, audio_processor, classifier, device_index=None):
        super().__init__()
        self.audio_processor = audio_processor
        self.classifier = classifier
        self.device_index = device_index
        self.running = False
        
    def run(self):
        self.running = True
        while self.running:
            try:
                audio_data = self.audio_processor.record_chunk(self.device_index)
                if audio_data is not None:
                    spectrogram = self.audio_processor.get_spectrogram_for_prediction(audio_data)
                    if spectrogram is not None:
                        result = self.classifier.predict(spectrogram)
                        if result:
                            self.result_ready.emit(result)
            except Exception as e:
                self.error_occurred.emit(str(e))
                
    def stop(self):
        self.running = False
        self.wait()

class ClassBar(QWidget):
    def __init__(self, label, label_de, emoji):
        super().__init__()
        self.label = label
        self.label_de = label_de
        self.emoji = emoji
        self.setup_ui()
        
    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 2, 10, 2)
        
        self.name_label = QLabel(f"> {self.label_de.upper()}")
        self.name_label.setFixedWidth(160)
        self.name_label.setStyleSheet(f"color: {THEME['accent_low']}; font-size: 11px;")
        
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(12)
        
        self.percent_label = QLabel("00.0%")
        self.percent_label.setFixedWidth(60)
        self.percent_label.setStyleSheet(f"font-size: 10px; color: {THEME['accent_low']};")
        
        layout.addWidget(self.name_label)
        layout.addWidget(self.progress, 1)
        layout.addWidget(self.percent_label)
        
    def set_value(self, probability, is_top=False):
        percent = probability * 100
        self.progress.setValue(int(percent))
        self.percent_label.setText(f"{percent:04.1f}%")
        
        if is_top:
            self.name_label.setStyleSheet(f"color: {THEME['accent']}; font-weight: bold;")
            self.percent_label.setStyleSheet(f"color: {THEME['accent']}; font-weight: bold;")
        else:
            self.name_label.setStyleSheet(f"color: {THEME['accent_low']};")
            self.percent_label.setStyleSheet(f"color: {THEME['accent_low']};")

class CrugerAcousticApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_processor = AudioProcessor()
        self.classifier = SoundClassifier()
        self.worker = None
        self.is_recording = False
        self.setup_ui()
        self.load_devices()
        
    def setup_ui(self):
        self.setWindowTitle("CRUGER DEFENSE - ACOUSTIC INTEL")
        self.setMinimumSize(800, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(20)
        layout.setContentsMargins(30, 30, 30, 30)
        
        # Header Section
        header_frame = QFrame()
        header_layout = QVBoxLayout(header_frame)
        
        header = QLabel("CRUGER // ACOUSTIC SCANNER")
        header.setFont(QFont("Consolas", 28, QFont.Weight.Bold))
        header.setStyleSheet(f"color: {THEME['accent']}; letter-spacing: 5px;")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        sub_header = QLabel("SYSTEM STATUS: READY_FOR_UPLINK")
        sub_header.setStyleSheet(f"color: {THEME['accent_low']}; letter-spacing: 2px;")
        sub_header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        header_layout.addWidget(header)
        header_layout.addWidget(sub_header)
        layout.addWidget(header_frame)

        # Tech Grid Layout
        main_grid = QHBoxLayout()
        
        # Left Side: Controls
        controls_panel = QVBoxLayout()
        
        model_group = QGroupBox("AI CORE CONFIG")
        model_layout = QVBoxLayout(model_group)
        self.model_path_label = QLabel("NO_CORE_LOADED")
        self.model_path_label.setStyleSheet("font-size: 10px; border-bottom: 1px solid #1a2a3a;")
        model_layout.addWidget(self.model_path_label)
        self.load_model_btn = QPushButton("UPLINK CORE")
        self.load_model_btn.clicked.connect(self.load_model)
        model_layout.addWidget(self.load_model_btn)
        controls_panel.addWidget(model_group)
        
        mic_group = QGroupBox("SENSOR INPUT")
        mic_layout = QVBoxLayout(mic_group)
        self.device_combo = QComboBox()
        mic_layout.addWidget(self.device_combo)
        self.refresh_btn = QPushButton("RESCAN DEVICES")
        self.refresh_btn.clicked.connect(self.load_devices)
        mic_layout.addWidget(self.refresh_btn)
        controls_panel.addWidget(mic_group)

        self.start_btn = QPushButton("ENGAGE ACOUSTIC SCAN")
        self.start_btn.setFont(QFont("Consolas", 12, QFont.Weight.Bold))
        self.start_btn.setFixedHeight(60)
        self.start_btn.setStyleSheet(f"border: 2px solid {THEME['accent']};")
        self.start_btn.clicked.connect(self.toggle_recording)
        self.start_btn.setEnabled(False)
        controls_panel.addWidget(self.start_btn)
        
        main_grid.addLayout(controls_panel, 1)

        # Right Side: Detection Display
        display_panel = QVBoxLayout()
        
        detection_group = QGroupBox("TARGET CLASSIFICATION")
        detection_layout = QVBoxLayout(detection_group)
        self.detected_label = QLabel("---")
        self.detected_label.setFont(QFont("Consolas", 32, QFont.Weight.Bold))
        self.detected_label.setStyleSheet(f"color: {THEME['accent']};")
        self.detected_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_layout.addWidget(self.detected_label)
        
        self.confidence_label = QLabel("CONFIDENCE: 0.0%")
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        detection_layout.addWidget(self.confidence_label)
        display_panel.addWidget(detection_group)
        
        bars_group = QGroupBox("SPECTRAL ANALYSIS")
        bars_layout = QVBoxLayout(bars_group)
        self.class_bars = {}
        for label in SoundClassifier.LABELS:
            label_de = SoundClassifier.LABELS_DE[label]
            emoji = SoundClassifier.LABEL_EMOJIS[label]
            bar = ClassBar(label, label_de, emoji)
            self.class_bars[label] = bar
            bars_layout.addWidget(bar)
        display_panel.addWidget(bars_group)
        
        main_grid.addLayout(display_panel, 2)
        layout.addLayout(main_grid)
        
        self.status_label = QLabel("SYSTEM_INIT_COMPLETE")
        self.status_label.setStyleSheet(f"font-size: 9px; color: {THEME['accent_low']};")
        layout.addWidget(self.status_label)

    def load_devices(self):
        self.device_combo.clear()
        devices = self.audio_processor.get_available_devices()
        for dev in devices:
            self.device_combo.addItem(f"CH_0{dev['index']} // {dev['name']}", dev["index"])
        self.status_label.setText(f"SENSORS_ACTIVE: {len(devices)}")
            
    def load_model(self):
        file_path, _ = QFileDialog.getOpenFileName()
        if file_path:
            try:
                self.classifier.load_model(file_path)
                self.model_path_label.setText(f"ACTIVE_CORE: {Path(file_path).name.upper()}")
                self.start_btn.setEnabled(True)
                self.status_label.setText("CORE_UPLINK_SUCCESSFUL")
            except Exception as e:
                QMessageBox.critical(self, "SYSTEM_ERROR", f"CORE_FAILURE: {e}")
                
    def toggle_recording(self):
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
            
    def start_recording(self):
        device_index = self.device_combo.currentData()
        self.worker = AudioWorker(self.audio_processor, self.classifier, device_index)
        self.worker.result_ready.connect(self.update_display)
        self.worker.error_occurred.connect(self.handle_error)
        self.worker.start()
        self.is_recording = True
        self.start_btn.setText("TERMINATE SCAN")
        self.start_btn.setStyleSheet(f"border: 2px solid {THEME['danger']}; color: {THEME['danger']};")
        self.status_label.setText("SCANNING_IN_PROGRESS...")
        
    def stop_recording(self):
        if self.worker:
            self.worker.stop()
            self.worker = None
        self.is_recording = False
        self.start_btn.setText("ENGAGE ACOUSTIC SCAN")
        self.start_btn.setStyleSheet(f"border: 2px solid {THEME['accent']}; color: {THEME['accent']};")
        self.status_label.setText("SCAN_HALTED")
        
    def update_display(self, result):
        label_de = result["label_de"]
        confidence = result["confidence"]
        self.detected_label.setText(f"{label_de.upper()}")
        self.confidence_label.setText(f"CONFIDENCE: {confidence*100:04.1f}%")
        
        top_label = result["label"]
        for label, prob in result["all_probs"].items():
            self.class_bars[label].set_value(prob, (label == top_label))
            
    def handle_error(self, error_msg):
        self.status_label.setText(f"CRITICAL_FAILURE: {error_msg}")
        
    def closeEvent(self, event):
        if self.worker:
            self.worker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = CrugerAcousticApp()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()