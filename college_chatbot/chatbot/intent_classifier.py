"""
Intent Classifier for the College Enquiry Chatbot.

Implements a hybrid classification approach:
1. Exact pattern matching
2. TF-IDF cosine similarity (via NLPEngine)
3. Keyword fallback rules
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path

from chatbot.nlp_engine import NLPEngine

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_FILE = LOG_DIR / "chat_log.txt"
INTENTS_PATH = BASE_DIR / "data" / "intents.json"

# Ensure log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)


class IntentClassifier:
    """
    Hybrid intent classifier combining exact match, TF-IDF cosine similarity,
    and keyword-based fallback rules.
    """

    def __init__(self, nlp_engine: NLPEngine):
        """
        Initialize the classifier.

        Args:
            nlp_engine: An initialized NLPEngine instance.
        """
        self.nlp = nlp_engine
        self.exact_patterns = self._build_exact_patterns()
        logger.info("IntentClassifier initialized with %d exact patterns.", len(self.exact_patterns))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str, session_id: str = "unknown") -> dict:
        """
        Classify user input into an intent using the hybrid approach.

        Args:
            text: Raw user message.
            session_id: Session identifier for logging.

        Returns:
            Dict with keys: intent, confidence, matched_keywords, tokens, method.
        """
        tokens, cleaned = self.nlp.preprocess(text)
        text_lower = text.lower().strip()

        # Step 1: Exact pattern match
        exact_result = self._exact_match(text_lower)
        if exact_result:
            result = {
                "intent": exact_result,
                "confidence": 1.0,
                "matched_keywords": [],
                "tokens": tokens,
                "method": "exact_match",
            }
            self._log(session_id, text, result)
            return result

        # Step 2: TF-IDF cosine similarity
        intent, confidence = self.nlp.predict_intent(text)

        # Step 3: Keyword fallback — if cosine gave fallback, try keyword rules
        if intent == "fallback":
            kw_intent = self._keyword_fallback(tokens)
            if kw_intent:
                intent = kw_intent
                confidence = 0.45  # moderate confidence for keyword fallback

        # Gather matched keywords for transparency
        kw_matches = self.nlp.get_keyword_matches(tokens)
        matched_keywords = kw_matches.get(intent, [])

        # Detect angry/frustrated user
        if self._is_angry(tokens):
            intent = "angry_user"
            confidence = 0.99

        result = {
            "intent": intent,
            "confidence": round(confidence, 4),
            "matched_keywords": matched_keywords,
            "tokens": tokens,
            "method": "tfidf_cosine" if intent != "fallback" else "fallback",
        }
        self._log(session_id, text, result)
        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_exact_patterns(self) -> dict[str, str]:
        """
        Build a dict mapping lowercase pattern strings to their intent tags.

        Returns:
            Dict of {pattern_string: intent_tag}.
        """
        mapping: dict[str, str] = {}
        with open(INTENTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        for intent in data["intents"]:
            for pattern in intent["patterns"]:
                mapping[pattern.lower().strip()] = intent["tag"]
        return mapping

    def _exact_match(self, text_lower: str) -> str | None:
        """
        Check whether the user text exactly matches a known pattern.

        Args:
            text_lower: Lowercased user input.

        Returns:
            Intent tag if matched, else None.
        """
        return self.exact_patterns.get(text_lower)

    def _keyword_fallback(self, tokens: list[str]) -> str | None:
        """
        Use keyword presence to guess intent when TF-IDF confidence is low.

        Args:
            tokens: Preprocessed tokens.

        Returns:
            Intent tag if a keyword is found, else None.
        """
        kw_matches = self.nlp.get_keyword_matches(tokens)
        if not kw_matches:
            return None
        # Return intent with most keyword matches
        return max(kw_matches, key=lambda k: len(kw_matches[k]))

    def _is_angry(self, tokens: list[str]) -> bool:
        """
        Detect frustrated or angry user messages.

        Args:
            tokens: Preprocessed tokens.

        Returns:
            True if anger indicators detected.
        """
        anger_words = {
            "useless", "worst", "pathetic", "terrible", "horrible",
            "stupid", "idiot", "waste", "rubbish", "garbage", "hate",
            "disgusting", "awful", "ridiculous", "nonsense"
        }
        return bool(anger_words.intersection(set(tokens)))

    def _log(self, session_id: str, user_text: str, result: dict) -> None:
        """
        Append classification result to the chat log file.

        Args:
            session_id: Session identifier.
            user_text: Original user input.
            result: Classification result dict.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = (
            f"[{timestamp}] session={session_id} | "
            f"intent={result['intent']} | "
            f"confidence={result['confidence']:.4f} | "
            f"method={result['method']} | "
            f"text={user_text!r}\n"
        )
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(log_line)
        except OSError as exc:
            logger.error("Failed to write to chat log: %s", exc)
