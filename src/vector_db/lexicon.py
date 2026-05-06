"""
Dynamic lexicon: maps 12-dim chroma vectors → English concept words.

Storage
-------
  • FAISS IndexFlatIP on L2-normalised vectors (dot product == cosine similarity)
  • Parallel list stores {word, chord_hint} metadata
  • Persisted as <path>.npy (vectors) + <path>.json (metadata)

Thresholds (from research docs)
--------------------------------
  cosine >= 0.85  →  high-confidence match
  cosine >= 0.70  →  probable match (flagged)
  cosine <  0.70  →  unknown — trigger UI prompt
"""

import json
import numpy as np
import faiss
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


MATCH_THRESHOLD = 0.70       # below this → "unknown"
HIGH_CONFIDENCE = 0.85


@dataclass
class LexiconEntry:
    word: str
    chord_hint: str = ''     # e.g. 'C_maj' — informational only


@dataclass
class SearchResult:
    word: Optional[str]
    similarity: float
    is_known: bool           # True if similarity >= MATCH_THRESHOLD
    is_confident: bool       # True if similarity >= HIGH_CONFIDENCE


class DynamicLexicon:
    def __init__(self):
        self._index = faiss.IndexFlatIP(12)   # inner product on unit vecs = cosine
        self._entries: list[LexiconEntry] = []

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def add_entry(self, chroma: np.ndarray, word: str, chord_hint: str = '') -> None:
        """Add or update a word mapping for the given chroma vector."""
        vec = self._normalise(chroma).reshape(1, 12)
        self._index.add(vec)
        self._entries.append(LexiconEntry(word=word, chord_hint=chord_hint))

    def search(self, chroma: np.ndarray) -> SearchResult:
        """Find the closest word for a chroma vector."""
        if self._index.ntotal == 0:
            return SearchResult(word=None, similarity=0.0,
                                is_known=False, is_confident=False)

        vec = self._normalise(chroma).reshape(1, 12)
        distances, indices = self._index.search(vec, k=1)
        sim = float(distances[0, 0])
        idx = int(indices[0, 0])

        if idx < 0 or idx >= len(self._entries):
            return SearchResult(word=None, similarity=sim,
                                is_known=False, is_confident=False)

        word = self._entries[idx].word
        return SearchResult(
            word=word,
            similarity=sim,
            is_known=sim >= MATCH_THRESHOLD,
            is_confident=sim >= HIGH_CONFIDENCE,
        )

    def size(self) -> int:
        return self._index.ntotal

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        vectors = faiss.rev_swig_ptr(self._index.get_xb(), self._index.ntotal * 12)
        np.save(str(p.with_suffix('.npy')), np.array(vectors).reshape(-1, 12))

        meta = [{'word': e.word, 'chord_hint': e.chord_hint} for e in self._entries]
        with open(p.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    def load(self, path: str) -> None:
        p = Path(path)
        npy_path = p.with_suffix('.npy')
        json_path = p.with_suffix('.json')

        if not npy_path.exists() or not json_path.exists():
            raise FileNotFoundError(f'Lexicon files not found at {path}')

        vectors = np.load(str(npy_path)).astype(np.float32)
        with open(json_path, 'r', encoding='utf-8') as f:
            meta = json.load(f)

        self._index = faiss.IndexFlatIP(12)
        self._index.add(vectors)
        self._entries = [LexiconEntry(**m) for m in meta]

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def all_words(self) -> list[str]:
        return [e.word for e in self._entries]

    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(chroma: np.ndarray) -> np.ndarray:
        v = chroma.astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-8 else v


# ---------------------------------------------------------------------------
# Factory: build a lexicon pre-seeded from a JSON vocabulary file
# ---------------------------------------------------------------------------

def load_default_lexicon(json_path: str) -> DynamicLexicon:
    """
    JSON format:
    [{"word": "Friend", "chord_hint": "C_maj", "chroma": [1,0,0,0,1,0,0,1,0,0,0,0]}, ...]
    """
    lexicon = DynamicLexicon()
    with open(json_path, 'r', encoding='utf-8') as f:
        entries = json.load(f)

    for entry in entries:
        chroma = np.array(entry['chroma'], dtype=np.float32)
        lexicon.add_entry(chroma, word=entry['word'],
                          chord_hint=entry.get('chord_hint', ''))
    return lexicon
