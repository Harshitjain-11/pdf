"""
setup.py — One-shot setup script for the College Enquiry Chatbot.

Performs:
1. pip install -r requirements.txt
2. Downloads required NLTK data packages
3. Initializes the SQLite database
4. Trains and saves the TF-IDF vectorizer model
5. Prints a success message
"""

import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def run(cmd: list[str], desc: str) -> None:
    """Run a subprocess command and exit on failure."""
    print(f"\n{'='*60}")
    print(f"  {desc}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] '{' '.join(cmd)}' failed with exit code {result.returncode}.")
        sys.exit(result.returncode)


def install_requirements() -> None:
    """Install Python dependencies from requirements.txt."""
    req_path = BASE_DIR / "requirements.txt"
    run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path)],
        "Installing Python dependencies…",
    )


def download_nltk_data() -> None:
    """Download all required NLTK corpora and models."""
    print(f"\n{'='*60}")
    print("  Downloading NLTK data packages…")
    print(f"{'='*60}")

    import nltk

    packages = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "averaged_perceptron_tagger",
        "averaged_perceptron_tagger_eng",
        "maxent_ne_chunker",
        "maxent_ne_chunker_tab",
        "words",
        "omw-1.4",
    ]
    for pkg in packages:
        print(f"  Downloading {pkg}…", end=" ", flush=True)
        try:
            nltk.download(pkg, quiet=True)
            print("✓")
        except Exception as exc:
            print(f"⚠ {exc}")


def init_database() -> None:
    """Initialize the SQLite database schema."""
    print(f"\n{'='*60}")
    print("  Initializing SQLite database…")
    print(f"{'='*60}")

    # Add college_chatbot to path so imports work
    sys.path.insert(0, str(BASE_DIR))
    from database.db_manager import init_db

    init_db()
    print("  Database initialized successfully. ✓")


def train_model() -> None:
    """Train and persist the TF-IDF vectorizer."""
    print(f"\n{'='*60}")
    print("  Training TF-IDF model…")
    print(f"{'='*60}")

    sys.path.insert(0, str(BASE_DIR))
    from chatbot.nlp_engine import NLPEngine

    engine = NLPEngine()  # Training happens automatically in __init__
    print(f"  Model trained on {len(engine.all_patterns)} patterns. ✓")


def main() -> None:
    print("\n🎓  Institute of Technology & Management, Gwalior — Chatbot Setup")
    print("=" * 60)

    install_requirements()
    download_nltk_data()
    init_database()
    train_model()

    print(f"\n{'='*60}")
    print("  ✅ Setup complete!")
    print("  Run the chatbot with:")
    print("      python app.py")
    print("  Then open: http://localhost:5000")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
