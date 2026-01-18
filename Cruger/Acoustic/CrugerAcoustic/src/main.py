#!/usr/bin/env python3
"""
Cruger Acoustic - Main Entry Point
Urban Sound Classification System
"""

import sys
import os

# Füge src-Verzeichnis zum Pfad hinzu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ui_main import main

if __name__ == "__main__":
    print("━" * 50)
    print("  CRUGER ACOUSTIC")
    print("  Urban Sound Classification System")
    print("━" * 50)
    print()
    print("  Starte Anwendung...")
    print()
    
    main()