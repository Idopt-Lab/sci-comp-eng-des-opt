"""Thin Streamlit entry point for the aircraft cost trade studies."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from aircraft_sizing.examples.ex_04_cost.viz.streamlit_page import render

render()
