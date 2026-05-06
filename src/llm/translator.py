"""
Eridian → English translation engine.

Uses the Anthropic Messages API with:
  - Prompt caching on the system persona (cache_control ephemeral)
  - Few-shot Chain-of-Thought examples injected via the system prompt
  - A rolling conversation buffer (last 10 exchanges) for context continuity
  - Graceful degradation: if the API key is missing, returns a raw join of words
"""

import os
import json
import logging
from collections import deque
from typing import Optional

import anthropic

log = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 512
BUFFER_SIZE = 10   # keep last N exchanges for context

# ---------------------------------------------------------------------------
# System prompt — cached so only charged once per 5-minute TTL
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are the Eridian Language Translator, the AI core of the first-contact \
translation device from Project Hail Mary.

Context
-------
Eridians communicate exclusively through polyphonic harmonic chords. \
Each unique chord (or chord combination) encodes a single concept-word. \
When they "speak", a sequence of chords plays; this system converts each chord \
to its mapped English concept word and passes the list to you.

Your task
---------
Given a raw list of concept words decoded from Eridian chord patterns, \
reconstruct the most natural, grammatically correct English sentence \
that captures the speaker's intent.

Rules
-----
1. Think step-by-step inside <think>…</think> tags (hidden from the user).
2. Eridians may use non-English word order (SOV, OVS, VSO …). Infer the correct structure.
3. Use prior conversation exchanges for context — pronouns and topics carry over.
4. If a word is genuinely ambiguous, briefly note the ambiguity in parentheses.
5. Output format: <think> reasoning </think> then a single translated sentence on its own line.
6. Never invent words that were not in the input list.
7. Keep translations concise — Eridians are efficient communicators.

Few-shot examples
-----------------
Input: ['Friend', 'Greet']
<think>Two words: Friend (C major) + Greet (G dominant 7). The Eridian is greeting someone as a friend.</think>
Hello, friend!

Input: ['Question', 'Food', 'Need']
<think>A question about needing food. Most natural English: "Do you need food?"</think>
Do you need food?

Input: ['Danger', 'Star', 'Many', 'Travel']
<think>Danger from many stars ahead — likely a navigation warning during space travel.</think>
There is danger ahead — many stars on our travel path.

Input: ['Friend', 'Question', 'Understand', 'Yes']
<think>The Eridian is confirming understanding: "Do you understand, friend? Yes."</think>
Do you understand, friend? Yes, I do.
"""


class EridianTranslator:
    def __init__(self, api_key: Optional[str] = None):
        key = api_key or os.getenv('ANTHROPIC_API_KEY', '')
        self._ready = bool(key)
        if self._ready:
            self._client = anthropic.Anthropic(api_key=key)
        else:
            log.warning('ANTHROPIC_API_KEY not set — translator will echo raw words.')
        self._buffer: deque[dict] = deque(maxlen=BUFFER_SIZE * 2)  # user+assistant pairs

    # ------------------------------------------------------------------
    def translate_keywords(self, words: list[str]) -> str:
        """
        Convert a list of concept words into a natural English sentence.
        Returns the sentence string.
        """
        if not words:
            return ''

        if not self._ready:
            return f'[Raw] {" | ".join(words)}'

        user_text = f"Detected Eridian words: {json.dumps(words)}"

        messages = list(self._buffer) + [{"role": "user", "content": user_text}]

        try:
            response = self._client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},   # prompt caching
                    }
                ],
                messages=messages,
            )

            raw = response.content[0].text.strip()
            sentence = self._extract_sentence(raw)

            # Update conversation buffer
            self._buffer.append({"role": "user", "content": user_text})
            self._buffer.append({"role": "assistant", "content": raw})

            return sentence

        except anthropic.APIError as exc:
            log.error('Anthropic API error: %s', exc)
            return f'[API Error] {" | ".join(words)}'

    # ------------------------------------------------------------------
    def clear_context(self) -> None:
        """Reset conversation memory (e.g. new session)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_sentence(raw: str) -> str:
        """Strip <think>…</think> block and return the final sentence."""
        import re
        cleaned = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        # Take the last non-empty line
        lines = [l.strip() for l in cleaned.splitlines() if l.strip()]
        return lines[-1] if lines else raw.strip()
