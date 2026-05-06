"""
Entry point for the Eridian Language Learning system.

Usage:
    python main.py           # real microphone mode
    python main.py --demo    # demo mode (synthetic alien audio, no mic required)
"""

import sys
import argparse
import logging

from dotenv import load_dotenv
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt

from src.ui.dashboard import MainWindow


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Eridian Language Learning — real-time chord-to-English translator'
    )
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode using synthetic alien audio (no microphone required)',
    )
    parser.add_argument(
        '--log-level',
        default='WARNING',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Set the logging verbosity (default: WARNING)',
    )
    return parser.parse_args()


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name),
        format='%(asctime)s  %(levelname)-8s  %(name)s — %(message)s',
        datefmt='%H:%M:%S',
    )


def main() -> None:
    load_dotenv()
    args = _parse_args()
    _setup_logging(args.log_level)

    app = QApplication(sys.argv)
    app.setApplicationName('Eridian Language Learning')
    app.setApplicationVersion('1.0.0')

    # High-DPI support (Qt6 handles scaling automatically, but be explicit)
    app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    window = MainWindow(demo_mode=args.demo)
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
