"""Problem 3 — menu entry in the shared OrbitLab Streamlit website."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from artemislab.streamlit_app import render
render()
