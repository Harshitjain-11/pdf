"""
NLP Engine for the College Enquiry Chatbot.

Provides preprocessing pipeline, TF-IDF vectorization,
cosine similarity matching, and keyword boosting.
"""

import os
import re
import json
import logging
import joblib
import numpy as np
from pathlib import Path

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INTENTS_PATH = BASE_DIR / "data" / "intents.json"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"


def _get_wordnet_pos(treebank_tag: str) -> str:
    """Convert Penn Treebank POS tag to WordNet POS constant."""
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    if treebank_tag.startswith("V"):
        return wordnet.VERB
    if treebank_tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


class NLPEngine:
    """
    Core NLP processing engine providing:
    - Text preprocessing pipeline (tokenization, stopword removal, lemmatization)
    - TF-IDF vectorization with bigrams
    - Cosine similarity intent matching
    - Keyword-based confidence boosting
    """

    # High-signal keywords per intent for confidence boosting
    KEYWORD_BOOST_MAP = {
        "greeting": ["hi", "hello", "namaste", "hey", "greet", "namaskar"],
        "goodbye": ["bye", "goodbye", "thanks", "thank", "dhanyawad", "alvida"],
        "admission_process": [
            "admission", "apply", "addmission", "enroll", "register",
            "process", "procedure", "kaise", "karein"
        ],
        "courses_offered": [
            "course", "courses", "corse", "program", "programs",
            "department", "study", "btech", "bca", "bba", "mba"
        ],
        "eligibility": [
            "eligib", "eligible", "marks", "percentage", "qualify",
            "qualification", "minimum", "criteria", "requirement", "cutoff"
        ],
        "fees_structure": [
            "fee", "fees", "fess", "cost", "charge", "charges",
            "kitni", "paise", "rupee", "inr", "annual", "semester", "tuition"
        ],
        "last_date": [
            "last", "deadline", "date", "closing", "close",
            "kab", "last date", "cutoff date", "registration close"
        ],
        "documents_needed": [
            "document", "documents", "docoments", "certificate", "paper",
            "papers", "marksheet", "aadhar", "photograph", "bring"
        ],
        "hostel_info": [
            "hostel", "accommodation", "room", "stay", "lodge",
            "boarding", "campus", "boys hostel", "girls hostel"
        ],
        "contact_info": [
            "contact", "phone", "email", "address", "reach",
            "call", "helpline", "number", "location", "map"
        ],
        "scholarship": [
            "scholarship", "scholership", "financial", "aid", "merit",
            "waiver", "discount", "stipend", "free", "ews", "sc", "st"
        ],
        "exam_schedule": [
            "exam", "examination", "test", "schedule", "date",
            "syllabus", "pattern", "sitee", "jee", "gate", "timetable"
        ],
        "result_status": [
            "result", "merit list", "selection", "selected",
            "cutoff", "waitlist", "rank", "score", "declared"
        ],
        "slot_booking": [
            "book", "appointment", "slot", "visit", "counselling",
            "counseling", "schedule", "meeting", "campus visit"
        ],
    }

    def __init__(self):
        """Initialize NLP engine: load intents and set up vectorizer."""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self.intents_data = self._load_intents()
        self.all_patterns, self.pattern_labels = self._collect_patterns()
        self.vectorizer, self.pattern_vectors = self._load_or_train_vectorizer()
        logger.info("NLPEngine initialized with %d intent patterns.", len(self.all_patterns))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def preprocess(self, text: str) -> tuple[list[str], str]:
        """
        Full preprocessing pipeline.

        Args:
            text: Raw user input string.

        Returns:
            Tuple of (list of lemmatized tokens, cleaned string joined by spaces).
        """
        # Lowercase
        text = text.lower()
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords (but keep short tokens useful for intent detection)
        filtered = [t for t in tokens if t not in self.stop_words or len(t) <= 2]
        # POS tagging
        pos_tags = pos_tag(filtered)
        # Lemmatize using POS
        lemmatized = [
            self.lemmatizer.lemmatize(word, pos=_get_wordnet_pos(tag))
            for word, tag in pos_tags
        ]
        cleaned_str = " ".join(lemmatized)
        return lemmatized, cleaned_str

    def predict_intent(self, text: str) -> tuple[str, float]:
        """
        Predict the intent of the input text using TF-IDF cosine similarity.

        Args:
            text: Raw user input string.

        Returns:
            Tuple of (intent_tag, confidence_score).
        """
        tokens, cleaned = self.preprocess(text)
        vec = self.vectorizer.transform([cleaned])
        similarities = cosine_similarity(vec, self.pattern_vectors).flatten()

        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        best_intent = self.pattern_labels[best_idx]

        # Apply keyword boost
        boosted_intent, boosted_score = self._apply_keyword_boost(
            tokens, best_intent, best_score, similarities
        )

        if boosted_score < 0.20:
            logger.debug("Low confidence %.3f — returning fallback.", boosted_score)
            return "fallback", boosted_score

        logger.debug("Predicted intent '%s' with confidence %.3f.", boosted_intent, boosted_score)
        return boosted_intent, min(boosted_score, 1.0)

    def get_keyword_matches(self, tokens: list[str]) -> dict[str, list[str]]:
        """
        Return a dict mapping each intent to its matched keywords from tokens.

        Args:
            tokens: List of preprocessed tokens.

        Returns:
            Dict mapping intent tag to list of matched keywords.
        """
        matches: dict[str, list[str]] = {}
        token_set = set(tokens)
        for intent, keywords in self.KEYWORD_BOOST_MAP.items():
            found = [kw for kw in keywords if kw in token_set]
            if found:
                matches[intent] = found
        return matches

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_intents(self) -> dict:
        """Load intents.json data."""
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    def _collect_patterns(self) -> tuple[list[str], list[str]]:
        """Collect all patterns and their corresponding intent labels."""
        patterns: list[str] = []
        labels: list[str] = []
        for intent in self.intents_data["intents"]:
            for pattern in intent["patterns"]:
                patterns.append(pattern.lower())
                labels.append(intent["tag"])
        return patterns, labels

    def _load_or_train_vectorizer(self) -> tuple[TfidfVectorizer, np.ndarray]:
        """
        Load vectorizer from disk if up-to-date, otherwise retrain and save.

        Returns:
            Tuple of (fitted TfidfVectorizer, pattern vector matrix).
        """
        needs_training = True
        if VECTORIZER_PATH.exists():
            vectorizer_mtime = VECTORIZER_PATH.stat().st_mtime
            intents_mtime = INTENTS_PATH.stat().st_mtime
            if vectorizer_mtime >= intents_mtime:
                try:
                    vectorizer = joblib.load(VECTORIZER_PATH)
                    vectors = vectorizer.transform(self.all_patterns)
                    logger.info("Loaded existing vectorizer from %s.", VECTORIZER_PATH)
                    needs_training = False
                    return vectorizer, vectors
                except Exception as exc:
                    logger.warning("Failed to load vectorizer: %s. Retraining.", exc)

        if needs_training:
            return self._train_and_save_vectorizer()

    def _train_and_save_vectorizer(self) -> tuple[TfidfVectorizer, np.ndarray]:
        """Train a TF-IDF vectorizer on all patterns and save to disk."""
        vectorizer = TfidfVectorizer(
            analyzer="word",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        vectors = vectorizer.fit_transform(self.all_patterns)
        VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)
        logger.info("Trained and saved vectorizer to %s.", VECTORIZER_PATH)
        return vectorizer, vectors

    def _apply_keyword_boost(
        self,
        tokens: list[str],
        best_intent: str,
        best_score: float,
        similarities: np.ndarray,
    ) -> tuple[str, float]:
        """
        Boost intent confidence when high-signal keywords are present.

        Args:
            tokens: Preprocessed tokens.
            best_intent: Currently best intent from cosine similarity.
            best_score: Current best confidence score.
            similarities: Full similarity vector over all patterns.

        Returns:
            Tuple of (possibly updated intent, possibly boosted score).
        """
        token_set = set(tokens)
        boosted_scores: dict[str, float] = {}

        for intent, keywords in self.KEYWORD_BOOST_MAP.items():
            if any(kw in token_set for kw in keywords):
                # Find max similarity among patterns for this intent
                intent_indices = [
                    i for i, lbl in enumerate(self.pattern_labels) if lbl == intent
                ]
                if intent_indices:
                    intent_score = float(np.max(similarities[intent_indices]))
                    boosted_scores[intent] = intent_score + 0.15

        if boosted_scores:
            top_boosted_intent = max(boosted_scores, key=boosted_scores.__getitem__)
            top_boosted_score = boosted_scores[top_boosted_intent]
            if top_boosted_score > best_score:
                return top_boosted_intent, top_boosted_score

        return best_intent, best_score
