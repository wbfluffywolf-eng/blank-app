"""Problem 1: shown alongside the Problem 2 app when OrbitLab is deployed."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from nozzlelab.streamlit_app import render
render()
