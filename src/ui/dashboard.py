"""
Eridian Translator — PyQt6 Real-time Dashboard

Layout
------
  ┌─────────────────────────────────────────────────────────┐
  │  Toolbar: [▶ Start] [■ Stop] [Demo] [Clear] [Add Word]  │
  ├─────────────────┬───────────────────────────────────────┤
  │                 │  RAW SIGNAL LOG                       │
  │  SPECTROGRAM    │  ─────────────────────────────────    │
  │  (PyQtGraph)    │  TRANSLATION CHAT                     │
  │                 │                                       │
  ├─────────────────┴───────────────────────────────────────┤
  │  Status bar: RMS | last chord | lexicon size            │
  └─────────────────────────────────────────────────────────┘

Threading model
---------------
  • AudioWorker (QThread)      – audio capture + chord detection
  • TranslationWorker (QThread) – Anthropic API calls
  • All GUI updates via pyqtSignal
"""

import queue
import numpy as np

import pyqtgraph as pg
from pyqtgraph import ColorMap

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QVBoxLayout, QHBoxLayout,
    QTextEdit, QToolBar, QStatusBar, QLabel, QDialog,
    QDialogButtonBox, QLineEdit, QInputDialog, QMessageBox,
    QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QColor, QTextCharFormat, QTextCursor, QFont, QAction

from src.audio.processor import N_FREQ_BINS
from src.vector_db.lexicon import DynamicLexicon, load_default_lexicon
from src.llm.translator import EridianTranslator
from src.ui.workers import AudioWorker, TranslationWorker, WORD_BATCH_SIZE

# Number of spectrogram columns to display (time axis width)
SPEC_COLS = 300
# Colormap: dark-blue → cyan → yellow (mimics "inferno" style)
_CMAP_COLORS = [
    (0.0,  [0,   0,   50,  255]),
    (0.3,  [20,  50,  150, 255]),
    (0.6,  [0,   200, 200, 255]),
    (1.0,  [255, 240, 50,  255]),
]


def _make_colormap() -> ColorMap:
    pos = [c[0] for c in _CMAP_COLORS]
    color = [c[1] for c in _CMAP_COLORS]
    return ColorMap(pos, color)


class MainWindow(QMainWindow):
    def __init__(self, lexicon_path: str = 'data/default_lexicon',
                 demo_mode: bool = False):
        super().__init__()
        self.setWindowTitle('Eridian Language Translator  |  Project Hail Mary')
        self.resize(1200, 720)

        # -- state --
        self._demo_mode = demo_mode
        self._word_buffer: list[str] = []
        self._last_chord: str = ''
        self._running = False
        self._demo_generator = None

        # -- data --
        self._lexicon = self._load_lexicon(lexicon_path)
        self._translator = EridianTranslator()

        # Spectrogram rolling buffer [freq_bins × time_cols]
        self._spec_buf = np.full((N_FREQ_BINS, SPEC_COLS), -80.0, dtype=np.float32)

        # -- workers (created but not started yet) --
        self._demo_queue: queue.Queue | None = None
        self._audio_worker: AudioWorker | None = None
        self._trans_worker = TranslationWorker(self._translator)
        self._trans_worker.translation_ready.connect(self._on_translation)
        self._trans_worker.start()

        # -- build UI --
        self._build_ui()
        self._status_bar_update('Ready — press Start or Demo to begin.')

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        root_layout.addWidget(self._build_toolbar())

        # Main splitter (spectrogram | logs)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_spectrogram_panel())
        splitter.addWidget(self._build_log_panel())
        splitter.setSizes([480, 680])
        root_layout.addWidget(splitter)

        # Status bar
        self._status_label = QLabel('Idle')
        self._chord_label = QLabel('Chord: —')
        self._rms_label = QLabel('RMS: 0.000')
        self._lex_label = QLabel(f'Lexicon: {self._lexicon.size()} words')

        sb = QStatusBar()
        sb.addWidget(self._status_label, 3)
        sb.addPermanentWidget(self._chord_label)
        sb.addPermanentWidget(self._rms_label)
        sb.addPermanentWidget(self._lex_label)
        self.setStatusBar(sb)

    def _build_toolbar(self) -> QToolBar:
        tb = QToolBar('Controls')
        tb.setMovable(False)

        self._act_start = QAction('▶  Start Mic', self)
        self._act_start.triggered.connect(self._on_start_mic)
        tb.addAction(self._act_start)

        self._act_demo = QAction('🛸  Demo Mode', self)
        self._act_demo.triggered.connect(self._on_start_demo)
        tb.addAction(self._act_demo)

        self._act_stop = QAction('■  Stop', self)
        self._act_stop.triggered.connect(self._on_stop)
        self._act_stop.setEnabled(False)
        tb.addAction(self._act_stop)

        tb.addSeparator()

        act_clear = QAction('🗑  Clear Logs', self)
        act_clear.triggered.connect(self._on_clear)
        tb.addAction(act_clear)

        act_add = QAction('＋  Add Word', self)
        act_add.triggered.connect(self._on_manual_add)
        tb.addAction(act_add)

        act_context = QAction('↺  Reset Context', self)
        act_context.triggered.connect(self._on_reset_context)
        tb.addAction(act_context)

        return tb

    def _build_spectrogram_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel('  Real-time Spectrogram')
        title.setStyleSheet('font-weight: bold; color: #00ccff; padding: 4px;')
        layout.addWidget(title)

        self._spec_widget = pg.GraphicsLayoutWidget()
        self._spec_widget.setBackground('#0a0a1a')
        layout.addWidget(self._spec_widget)

        plot = self._spec_widget.addPlot()
        plot.setLabel('left', 'Frequency Bin')
        plot.setLabel('bottom', 'Time →')
        plot.showGrid(x=False, y=True, alpha=0.2)

        self._img_item = pg.ImageItem()
        self._img_item.setOpts(axisOrder='col-major')
        plot.addItem(self._img_item)

        cmap = _make_colormap()
        lut = cmap.getLookupTable(0.0, 1.0, 256)
        self._img_item.setLookupTable(lut)
        self._img_item.setLevels([-80, 0])

        self._img_item.setImage(self._spec_buf)

        # Chroma bar chart (12 pitch classes)
        self._chroma_widget = pg.PlotWidget(background='#0a0a1a')
        self._chroma_widget.setMaximumHeight(100)
        self._chroma_widget.setLabel('bottom', 'Pitch Class (C … B)')
        self._chroma_widget.showGrid(x=False, y=True, alpha=0.3)
        self._chroma_bar = pg.BarGraphItem(
            x=list(range(12)),
            height=[0] * 12,
            width=0.7,
            brush='#00ccff',
        )
        self._chroma_widget.addItem(self._chroma_bar)
        layout.addWidget(self._chroma_widget)

        return panel

    def _build_log_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 0, 0, 0)

        # Raw signal log
        raw_title = QLabel('  Raw Signal Log')
        raw_title.setStyleSheet('font-weight: bold; color: #ffaa00; padding: 4px;')
        layout.addWidget(raw_title)

        self._raw_log = QTextEdit()
        self._raw_log.setReadOnly(True)
        self._raw_log.setFont(QFont('Consolas', 9))
        self._raw_log.setStyleSheet(
            'background:#0f0f22; color:#cccccc; border:1px solid #333;'
        )
        self._raw_log.setMaximumHeight(200)
        layout.addWidget(self._raw_log)

        # Translation chat
        chat_title = QLabel('  Translation Chat')
        chat_title.setStyleSheet('font-weight: bold; color: #88ff88; padding: 4px;')
        layout.addWidget(chat_title)

        self._chat_log = QTextEdit()
        self._chat_log.setReadOnly(True)
        self._chat_log.setFont(QFont('Segoe UI', 10))
        self._chat_log.setStyleSheet(
            'background:#0f1a0f; color:#ddffdd; border:1px solid #336633;'
        )
        layout.addWidget(self._chat_log)

        # Pending words display
        pending_title = QLabel('  Pending Words (buffering → LLM)')
        pending_title.setStyleSheet('color: #888888; padding: 2px 4px;')
        layout.addWidget(pending_title)

        self._pending_label = QLabel('[ ]')
        self._pending_label.setStyleSheet(
            'color:#ffcc66; font-family:Consolas; padding:4px; '
            'background:#111100; border:1px solid #444;'
        )
        layout.addWidget(self._pending_label)

        return panel

    # ------------------------------------------------------------------
    # Toolbar slots
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_start_mic(self) -> None:
        if self._running:
            return
        self._start_audio_worker(demo=False)

    @pyqtSlot()
    def _on_start_demo(self) -> None:
        if self._running:
            return
        self._start_audio_worker(demo=True)

    def _start_audio_worker(self, demo: bool) -> None:
        self._demo_mode = demo
        if demo:
            from src.audio.demo_generator import DemoAudioGenerator
            self._demo_queue = queue.Queue(maxsize=32)
            self._demo_generator = DemoAudioGenerator(self._demo_queue)
            self._demo_generator.start()
            self._log_chat('[System]', 'Demo mode: synthetic Eridian signals playing…', '#888888')
        else:
            self._demo_queue = None
            self._demo_generator = None

        self._audio_worker = AudioWorker(self._lexicon, self._demo_queue)
        self._audio_worker.spec_update.connect(self._on_spec_update)
        self._audio_worker.word_detected.connect(self._on_word_detected)
        self._audio_worker.unknown_chord.connect(self._on_unknown_chord)
        self._audio_worker.rms_update.connect(self._on_rms_update)
        self._audio_worker.start()

        self._running = True
        self._act_start.setEnabled(False)
        self._act_demo.setEnabled(False)
        self._act_stop.setEnabled(True)
        mode = 'Demo' if demo else 'Microphone'
        self._status_bar_update(f'{mode} active — listening…')

    @pyqtSlot()
    def _on_stop(self) -> None:
        if self._demo_generator:
            self._demo_generator.stop()
            self._demo_generator = None
        if self._audio_worker:
            self._audio_worker.stop()
            self._audio_worker = None
        self._running = False
        self._act_start.setEnabled(True)
        self._act_demo.setEnabled(True)
        self._act_stop.setEnabled(False)
        self._status_bar_update('Stopped.')

    @pyqtSlot()
    def _on_clear(self) -> None:
        self._raw_log.clear()
        self._word_buffer.clear()
        self._update_pending_label()

    @pyqtSlot()
    def _on_reset_context(self) -> None:
        self._translator.clear_context()
        self._word_buffer.clear()
        self._update_pending_label()
        self._log_chat('[System]', 'Conversation context reset.', '#888888')

    @pyqtSlot()
    def _on_manual_add(self) -> None:
        """Let the user manually type a chord→word mapping using a chord name."""
        from src.audio.chord_templates import CHORD_TEMPLATES
        chord_name, ok1 = QInputDialog.getItem(
            self, 'Add Word — Step 1', 'Select chord:',
            sorted(CHORD_TEMPLATES.keys()), editable=False,
        )
        if not ok1:
            return
        word, ok2 = QInputDialog.getText(
            self, 'Add Word — Step 2', f'English word for chord "{chord_name}":',
        )
        if not ok2 or not word.strip():
            return
        chroma = CHORD_TEMPLATES[chord_name]
        self._lexicon.add_entry(chroma, word.strip(), chord_hint=chord_name)
        self._lex_label.setText(f'Lexicon: {self._lexicon.size()} words')
        self._log_raw(f'[Lexicon] Added: {chord_name} → "{word.strip()}"', '#aaaaff')

    # ------------------------------------------------------------------
    # Worker signal handlers
    # ------------------------------------------------------------------

    @pyqtSlot(object)
    def _on_spec_update(self, col: np.ndarray) -> None:
        self._spec_buf = np.roll(self._spec_buf, -1, axis=1)
        col_clipped = np.clip(col[:N_FREQ_BINS], -80, 0)
        self._spec_buf[:, -1] = col_clipped
        self._img_item.setImage(self._spec_buf, autoLevels=False)

    @pyqtSlot(str, float, str)
    def _on_word_detected(self, word: str, similarity: float, chord: str) -> None:
        confidence = '✓✓' if similarity >= 0.85 else '✓'
        line = f'{confidence} {chord:12s} → "{word}"  (sim={similarity:.3f})'
        color = '#00ff88' if similarity >= 0.85 else '#ffcc44'
        self._log_raw(line, color)

        self._chord_label.setText(f'Chord: {chord}')
        self._last_chord = chord

        # Update chroma bar from template
        from src.audio.chord_templates import CHORD_TEMPLATES
        if chord in CHORD_TEMPLATES:
            self._chroma_bar.setOpts(height=CHORD_TEMPLATES[chord].tolist())

        self._word_buffer.append(word)
        self._update_pending_label()

        if len(self._word_buffer) >= WORD_BATCH_SIZE:
            batch = list(self._word_buffer)
            self._word_buffer.clear()
            self._update_pending_label()
            self._log_raw(f'→ Sending to LLM: {batch}', '#8888ff')
            self._trans_worker.submit_words(batch)

    @pyqtSlot(object)
    def _on_unknown_chord(self, chroma: np.ndarray) -> None:
        self._log_raw('[?] Unknown chord — assign a word:', '#ff8844')

        word, ok = QInputDialog.getText(
            self, 'New Eridian Word',
            'Unknown chord detected.\nEnter the English concept word for this chord:',
        )
        if ok and word.strip():
            self._lexicon.add_entry(chroma, word.strip(), chord_hint='user-defined')
            self._lex_label.setText(f'Lexicon: {self._lexicon.size()} words')
            self._log_raw(f'[Lexicon] Learned: → "{word.strip()}"', '#aaaaff')
        else:
            self._log_raw('[?] Skipped — chord not added to lexicon.', '#888888')

    @pyqtSlot(str)
    def _on_translation(self, sentence: str) -> None:
        self._log_chat('Eridian', sentence, '#88ff88')

    @pyqtSlot(float)
    def _on_rms_update(self, rms: float) -> None:
        self._rms_label.setText(f'RMS: {rms:.4f}')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _log_raw(self, text: str, color: str = '#cccccc') -> None:
        self._raw_log.moveCursor(QTextCursor.MoveOperation.End)
        fmt = QTextCharFormat()
        fmt.setForeground(QColor(color))
        cursor = self._raw_log.textCursor()
        cursor.insertText(text + '\n', fmt)
        self._raw_log.setTextCursor(cursor)
        self._raw_log.ensureCursorVisible()

    def _log_chat(self, speaker: str, text: str, color: str = '#ddffdd') -> None:
        self._chat_log.moveCursor(QTextCursor.MoveOperation.End)
        cursor = self._chat_log.textCursor()

        # Speaker label
        label_fmt = QTextCharFormat()
        label_fmt.setForeground(QColor('#aaccff'))
        label_fmt.setFontWeight(700)
        cursor.insertText(f'{speaker}: ', label_fmt)

        # Message
        msg_fmt = QTextCharFormat()
        msg_fmt.setForeground(QColor(color))
        cursor.insertText(text + '\n\n', msg_fmt)

        self._chat_log.setTextCursor(cursor)
        self._chat_log.ensureCursorVisible()

    def _update_pending_label(self) -> None:
        if self._word_buffer:
            self._pending_label.setText('[ ' + ' | '.join(self._word_buffer) + ' ]')
        else:
            self._pending_label.setText('[ ]')

    def _status_bar_update(self, msg: str) -> None:
        self._status_label.setText(msg)

    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        self._on_stop()
        self._trans_worker.stop()
        event.accept()

    # ------------------------------------------------------------------
    @staticmethod
    def _load_lexicon(path: str) -> DynamicLexicon:
        try:
            return load_default_lexicon(f'{path}.json')
        except Exception:
            try:
                return load_default_lexicon('data/default_lexicon.json')
            except Exception:
                return DynamicLexicon()
