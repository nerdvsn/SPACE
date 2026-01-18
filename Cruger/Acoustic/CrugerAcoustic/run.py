#!/usr/bin/env python3
"""
Cruger Acoustic - Launcher
"""

import sys
import os

# Pfad zu src hinzufügen
root_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(root_dir, "src")
sys.path.insert(0, src_dir)

# Prüfe Dependencies
def check_dependencies():
    missing = []
    
    try:
        import PyQt6
    except ImportError:
        missing.append("PyQt6")
    
    try:
        import sounddevice
    except ImportError:
        missing.append("sounddevice")
    
    try:
        import librosa
    except ImportError:
        missing.append("librosa")
    
    try:
        import fastai
    except ImportError:
        missing.append("fastai")
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    if missing:
        print("=" * 50)
        print("  FEHLENDE PAKETE:")
        print("=" * 50)
        for pkg in missing:
            print(f"  - {pkg}")
        print()
        print("  Installiere mit:")
        print("  pip install -r requirements.txt")
        print("=" * 50)
        sys.exit(1)

if __name__ == "__main__":
    print()
    print("  ╔═══════════════════════════════════════════╗")
    print("  ║         CRUGER ACOUSTIC v1.0.0            ║")
    print("  ║   Urban Sound Classification System       ║")
    print("  ╚═══════════════════════════════════════════╝")
    print()
    
    print("  [1/2] Prüfe Abhängigkeiten...")
    check_dependencies()
    print("        ✓ Alle Pakete gefunden")
    print()
    
    print("  [2/2] Starte UI...")
    print()
    
    from ui_main import main
    main()