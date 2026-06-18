"""sign_rt — core library for real-time sign-language recognition & translation.

Single source of truth shared by the edge runner, FastAPI backend, and browser
demo: pose normalization, the graph backbone, the Conformer+CTC recognizer, the
translation decoder, and the streaming beam search.
"""
from __future__ import annotations

__version__ = "0.0.1"
