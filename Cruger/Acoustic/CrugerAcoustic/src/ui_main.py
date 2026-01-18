"""
CRUGER DEFENSE - MK V "FINAL STABILITY"
Fixed: Event Log Visibility, QPlainTextEdit Migration, Layout Optimization
"""

import sys
import numpy as np
from pathlib import Path
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QComboBox, QPushButton, QProgressBar, QFileDialog,
    QFrame, QGroupBox, QPlainTextEdit
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QDateTime, QPointF
from PyQt6.QtGui import QFont, QColor, QPainter, QPen, QBrush, QConicalGradient, QTextCursor

from audio_processor import AudioProcessor
from classifier import SoundClassifier

# --- THEME ---
THEME = {
    "bg": "#05080a",
    "panel": "#0a1218",
    "accent": "#00f0ff",
    "accent_low": "#004e52",
    "danger": "#ff0033",
    "radar_green": "#00ff41",
    "text": "#d1f2f7"
}

DANGER_SIGNATURES = ["gun_shot", "siren", "drilling"]

class AudioWorker(QThread):
    result_ready = pyqtSignal(dict)
    raw_data_ready = pyqtSignal(np.ndarray)
    
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
                    self.raw_data_ready.emit(audio_data)
                    spectrogram = self.audio_processor.get_spectrogram_for_prediction(audio_data)
                    if spectrogram is not None:
                        result = self.classifier.predict(spectrogram)
                        if result: self.result_ready.emit(result)
            except Exception: pass

    def stop(self):
        self.running = False
        self.wait()

class CircularRadar(QWidget):
    def __init__(self):
        super().__init__()
        self.angle = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_angle)
        self.timer.start(30)
        self.setMinimumSize(250, 250)

    def update_angle(self):
        self.angle = (self.angle + 3.0) % 360.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect().adjusted(20, 20, -20, -20)
        center = QPointF(rect.center())
        radius = rect.width() / 2.0

        painter.setPen(QPen(QColor(THEME['radar_green']), 1, Qt.PenStyle.DotLine))
        for r_f in [1.0, 0.6, 0.3]:
            painter.drawEllipse(center, radius * r_f, radius * r_f)

        gradient = QConicalGradient(center, -self.angle)
        gradient.setColorAt(0, QColor(0, 255, 65, 180))
        gradient.setColorAt(0.1, QColor(0, 255, 65, 50))
        gradient.setColorAt(0.3, QColor(0, 255, 65, 0))
        
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPie(rect, int(self.angle * 16), int(45 * 16))
        painter.end()

class LiveWaveform(pg.PlotWidget):
    def __init__(self, buffer_size=4000):
        super().__init__()
        self.buffer_size = buffer_size
        self.setBackground(THEME['bg'])
        self.hideAxis('bottom')
        self.hideAxis('left')
        self.pen = pg.mkPen(color=THEME['radar_green'], width=2)
        self.curve = self.plot(pen=self.pen)
        self.setYRange(-0.5, 0.5) 
        self.setXRange(0, buffer_size)
        self.data = np.zeros(buffer_size)
        
    def update_plot(self, new_data):
        if len(new_data) > self.buffer_size:
            new_data = new_data[:self.buffer_size]
        self.data = np.roll(self.data, -len(new_data))
        self.data[-len(new_data):] = new_data
        self.curve.setData(self.data)

class ClassBar(QWidget):
    def __init__(self, label_de):
        super().__init__()
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        self.lbl = QLabel(label_de.upper())
        self.lbl.setFixedWidth(130)
        self.bar = QProgressBar()
        self.bar.setFixedHeight(6)
        self.bar.setTextVisible(False)
        layout.addWidget(self.lbl)
        layout.addWidget(self.bar)

    def update_state(self, val, active, danger):
        self.bar.setValue(int(val * 100))
        color = THEME['danger'] if danger else (THEME['accent'] if active else THEME['accent_low'])
        self.lbl.setStyleSheet(f"color: {color}; font-size: 10px; font-weight: {'bold' if active else 'normal'};")

class CrugerAcousticApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.audio_processor = AudioProcessor()
        self.classifier = SoundClassifier()
        self.worker = None
        
        self.setup_ui()
        self.apply_styles(THEME['accent'])
        self.load_devices()

        self.alarm_timer = QTimer()
        self.alarm_timer.timeout.connect(self.flash_ui)
        self.is_recording = False
        self.alarm_active = False
        self.flash_state = False

    def setup_ui(self):
        self.setWindowTitle("CRUGER DEFENSE MK-V")
        self.setMinimumSize(1250, 850)
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        header = QHBoxLayout()
        self.title_lbl = QLabel("TACTICAL ACOUSTIC SCANNER")
        self.title_lbl.setFont(QFont("Consolas", 22, QFont.Weight.Bold))
        header.addWidget(self.title_lbl)
        self.status_tag = QLabel("[ SYSTEM_IDLE ]")
        header.addWidget(self.status_tag, 0, Qt.AlignmentFlag.AlignRight)
        main_layout.addLayout(header)

        hud = QHBoxLayout()

        # Radar
        left = QVBoxLayout()
        radar_group = QGroupBox("SCANNER SONAR")
        rl = QVBoxLayout(radar_group)
        self.radar = CircularRadar()
        rl.addWidget(self.radar)
        left.addWidget(radar_group)

        ctrl = QGroupBox("CONTROLS")
        cl = QVBoxLayout(ctrl)
        self.dev_box = QComboBox()
        self.load_btn = QPushButton("UPLOAD AI CORE")
        self.load_btn.clicked.connect(self.load_model)
        self.scan_btn = QPushButton("ENGAGE SCAN")
        self.scan_btn.setFixedHeight(50)
        self.scan_btn.setEnabled(False)
        self.scan_btn.clicked.connect(self.toggle_scan)
        cl.addWidget(self.dev_box)
        cl.addWidget(self.load_btn)
        cl.addWidget(self.scan_btn)
        left.addWidget(ctrl)
        hud.addLayout(left, 1)

        # Mid
        mid = QVBoxLayout()
        wave_group = QGroupBox("SIGNAL WAVEFORM")
        wl = QVBoxLayout(wave_group)
        self.waveform = LiveWaveform()
        wl.addWidget(self.waveform)
        mid.addWidget(wave_group, 2)

        det_group = QGroupBox("CLASSIFICATION DATA")
        dl = QVBoxLayout(det_group)
        self.res_lbl = QLabel("READY TO SCAN...")
        self.res_lbl.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        self.conf_lbl = QLabel("CONFIDENCE: 0%")
        dl.addWidget(self.res_lbl)
        dl.addWidget(self.conf_lbl)
        mid.addWidget(det_group, 1)
        hud.addLayout(mid, 2)

        # Right (Log & Probabilities)
        right = QVBoxLayout()
        matrix_group = QGroupBox("PROBABILITY MATRIX")
        ml = QVBoxLayout(matrix_group)
        self.bars = {}
        for k in SoundClassifier.LABELS:
            b = ClassBar(SoundClassifier.LABELS_DE[k])
            self.bars[k] = b
            ml.addWidget(b)
        right.addWidget(matrix_group, 2)

        log_group = QGroupBox("EVENT LOG")
        ll = QVBoxLayout(log_group)
        # NEU: QPlainTextEdit statt QLabel für bessere Sichtbarkeit
        self.log_widget = QPlainTextEdit()
        self.log_widget.setReadOnly(True)
        self.log_widget.setFont(QFont("Consolas", 9))
        self.log_widget.setStyleSheet(f"background: transparent; color: {THEME['text']}; border: none;")
        ll.addWidget(self.log_widget)
        right.addWidget(log_group, 1)

        hud.addLayout(right, 1)
        main_layout.addLayout(hud)

    def apply_styles(self, col):
        self.setStyleSheet(f"""
            QMainWindow {{ background: {THEME['bg']}; }}
            QGroupBox {{ color: {col}; border: 1px solid {col}; background: {THEME['panel']}; font-family: 'Consolas'; margin-top: 10px; }}
            QLabel {{ color: {THEME['text']}; font-family: 'Consolas'; }}
            QPushButton {{ color: {col}; border: 1px solid {col}; padding: 8px; font-weight: bold; }}
            QPushButton:hover {{ background: {col}22; }}
            QProgressBar {{ border: 1px solid {col}; background: #000; }}
            QProgressBar::chunk {{ background: {col}; }}
            QComboBox {{ background: #000; color: {col}; border: 1px solid {col}; }}
        """)

    def load_devices(self):
        for d in self.audio_processor.get_available_devices():
            self.dev_box.addItem(f"CH_{d['index']}: {d['name'][:25]}", d['index'])

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName()
        if path:
            self.classifier.load_model(path)
            self.scan_btn.setEnabled(True)
            self.log("CORE_CONNECTED")

    def toggle_scan(self):
        if not self.is_recording:
            self.worker = AudioWorker(self.audio_processor, self.classifier, self.dev_box.currentData())
            self.worker.raw_data_ready.emit(np.zeros(4000)) # Init wave
            self.worker.raw_data_ready.connect(self.waveform.update_plot)
            self.worker.result_ready.connect(self.update_res)
            self.worker.start()
            self.scan_btn.setText("TERMINATE")
            self.status_tag.setText("[ SCANNING... ]")
            self.log("SCAN_ENGAGED")
            self.is_recording = True
        else:
            if self.worker: self.worker.stop()
            self.stop_alarm()
            self.is_recording = False
            self.scan_btn.setText("ENGAGE SCAN")
            self.log("SCAN_TERMINATED")

    def update_res(self, res):
        label = res['label']
        self.res_lbl.setText(res['label_de'].upper())
        self.conf_lbl.setText(f"CONFIDENCE: {res['confidence']*100:.1f}%")
        
        is_danger = label in DANGER_SIGNATURES and res['confidence'] > 0.4
        if is_danger: 
            self.start_alarm()
            print('\a') 
        else: 
            self.stop_alarm()

        for k, v in res['all_probs'].items():
            self.bars[k].update_state(v, k == label, k in DANGER_SIGNATURES and v > 0.4)

    def start_alarm(self):
        if not self.alarm_active:
            self.alarm_active = True
            self.alarm_timer.start(200)
            self.log(f"ALERT: {self.res_lbl.text()}")

    def stop_alarm(self):
        if self.alarm_active:
            self.alarm_active = False
            self.alarm_timer.stop()
            self.apply_styles(THEME['accent'])

    def flash_ui(self):
        c = THEME['danger'] if self.flash_state else "#330000"
        self.apply_styles(c)
        self.flash_state = not self.flash_state

    def log(self, msg):
        t = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.log_widget.appendPlainText(f"[{t}] {msg}")
        self.log_widget.moveCursor(QTextCursor.MoveOperation.End)

    def closeEvent(self, event):
        if self.worker: self.worker.stop()
        event.accept()

def main():
    app = QApplication(sys.argv)
    win = CrugerAcousticApp()
    win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()