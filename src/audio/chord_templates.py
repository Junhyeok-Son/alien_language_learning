"""
12-bin chord templates and cosine-similarity matcher.

Pitch class mapping: C=0, C#=1, D=2, D#=3, E=4, F=5, F#=6, G=7, G#=8, A=9, A#=10, B=11
"""

import numpy as np
from typing import Dict, Tuple

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Interval sets (semitones from root)
_INTERVALS = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'dom7': [0, 4, 7, 10],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],
}


def _build_template(root: int, intervals: list[int]) -> np.ndarray:
    vec = np.zeros(12, dtype=np.float32)
    for iv in intervals:
        vec[(root + iv) % 12] = 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


# Build all templates at import time
CHORD_TEMPLATES: Dict[str, np.ndarray] = {}
for _root, _note in enumerate(NOTE_NAMES):
    for _quality, _ivs in _INTERVALS.items():
        CHORD_TEMPLATES[f'{_note}_{_quality}'] = _build_template(_root, _ivs)

# Stack into a matrix for fast batch cosine similarity (shape: [N, 12])
_TEMPLATE_NAMES = list(CHORD_TEMPLATES.keys())
_TEMPLATE_MATRIX = np.stack([CHORD_TEMPLATES[n] for n in _TEMPLATE_NAMES], axis=0)  # [N, 12]


def match_chord_template(chroma: np.ndarray) -> Tuple[str, float]:
    """Return (chord_name, cosine_similarity) for the closest template."""
    norm = np.linalg.norm(chroma)
    if norm < 1e-8:
        return 'Silence', 0.0
    chroma_unit = chroma.astype(np.float32) / norm
    sims = _TEMPLATE_MATRIX @ chroma_unit          # [N]
    idx = int(np.argmax(sims))
    return _TEMPLATE_NAMES[idx], float(sims[idx])


def all_template_names() -> list[str]:
    return list(_TEMPLATE_NAMES)
