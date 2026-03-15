"""
Context Manager for the College Enquiry Chatbot.

Handles per-session multi-turn conversation state including:
- Conversation history (last 10 turns)
- Current and previous intent tracking
- Entity carry-over across turns
- Pronoun / anaphora resolution
"""

import re
import logging
from collections import deque
from datetime import datetime

logger = logging.getLogger(__name__)

# Pronouns and vague references that should be resolved to the last entity
PRONOUN_MAP = {
    "uska", "iska", "its", "that course", "wala", "wali",
    "same course", "that", "this", "it", "us", "their",
    "is course", "us course", "ye", "yeh",
    "uski", "iski", "wahi", "same", "usi ka", "usi ki"
}

MAX_HISTORY = 10  # keep last N turns per session


class ContextManager:
    """
    Manages per-session conversation context for multi-turn dialogue.

    State is stored in an in-memory dict keyed by session_id.
    """

    def __init__(self):
        """Initialize with empty session store."""
        # session_id -> context dict
        self._sessions: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_or_create(self, session_id: str) -> dict:
        """
        Get existing context for session_id or create a new one.

        Args:
            session_id: Unique session identifier.

        Returns:
            Context dict for the session.
        """
        if session_id not in self._sessions:
            self._sessions[session_id] = self._new_context()
            logger.debug("Created new context for session %s.", session_id)
        return self._sessions[session_id]

    def update(
        self,
        session_id: str,
        user_message: str,
        bot_response: str,
        intent: str,
        entities: dict,
        slot_state: str = "IDLE",
    ) -> None:
        """
        Record a completed conversation turn and update running context.

        Args:
            session_id: Unique session identifier.
            user_message: Raw user input.
            bot_response: Generated bot response.
            intent: Classified intent tag.
            entities: Extracted entities for this turn.
            slot_state: Current slot-filling state.
        """
        ctx = self.get_or_create(session_id)

        # Record turn in history (ring buffer)
        ctx["history"].append({
            "user": user_message,
            "bot": bot_response,
            "intent": intent,
            "entities": entities,
            "timestamp": datetime.utcnow().isoformat(),
        })

        # Update intent tracking
        ctx["previous_intent"] = ctx["current_intent"]
        ctx["current_intent"] = intent

        # Merge entities (non-None values override previous)
        for key, value in entities.items():
            if value is not None:
                ctx["entities_so_far"][key] = value

        # Track last course separately for pronoun resolution
        if entities.get("course"):
            ctx["last_course"] = entities["course"]

        ctx["slot_state"] = slot_state
        ctx["turn_count"] += 1

    def resolve_text(self, session_id: str, text: str) -> str:
        """
        Resolve pronouns and vague references using previous context.

        If the user says "uska fee?" after discussing B.Tech CS, this
        method returns "B.Tech CS fee?" so downstream processing works.

        Args:
            session_id: Unique session identifier.
            text: Raw user input for the current turn.

        Returns:
            Resolved text string (may be unchanged).
        """
        ctx = self.get_or_create(session_id)
        text_lower = text.lower().strip()

        # Check if any pronoun is present (use word-boundary-aware check)
        # Sort by length descending so longer matches are attempted first
        sorted_pronouns = sorted(PRONOUN_MAP, key=len, reverse=True)
        has_pronoun = any(pronoun in text_lower for pronoun in sorted_pronouns)
        if not has_pronoun:
            return text

        last_course = ctx.get("last_course") or ctx["entities_so_far"].get("course")
        if last_course:
            # Replace pronoun with the last mentioned course
            for pronoun in sorted_pronouns:
                if pronoun in text_lower:
                    # Try word-boundary replacement first
                    new_text = re.sub(
                        rf"\b{re.escape(pronoun)}\b",
                        last_course,
                        text,
                        flags=re.IGNORECASE,
                    )
                    if new_text != text:
                        text = new_text
                        logger.debug("Resolved '%s' → '%s' in: %r", pronoun, last_course, text)
                        break

        return text

    def get_previous_intent(self, session_id: str) -> str | None:
        """Return the previous intent for the session."""
        ctx = self.get_or_create(session_id)
        return ctx.get("previous_intent")

    def get_entities_so_far(self, session_id: str) -> dict:
        """Return all accumulated entities for the session."""
        ctx = self.get_or_create(session_id)
        return dict(ctx["entities_so_far"])

    def get_history(self, session_id: str) -> list[dict]:
        """Return conversation history as a list of turn dicts."""
        ctx = self.get_or_create(session_id)
        return list(ctx["history"])

    def clear(self, session_id: str) -> None:
        """Remove all context for the session."""
        self._sessions.pop(session_id, None)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _new_context(self) -> dict:
        """Create and return a fresh context dict."""
        return {
            "history": deque(maxlen=MAX_HISTORY),
            "current_intent": None,
            "previous_intent": None,
            "slot_state": "IDLE",
            "entities_so_far": {},
            "last_course": None,
            "turn_count": 0,
            "created_at": datetime.utcnow().isoformat(),
        }
