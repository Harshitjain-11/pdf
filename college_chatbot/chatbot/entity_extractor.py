"""
Entity Extractor for the College Enquiry Chatbot.

Extracts named entities from user input:
- Person name (NLTK NE chunk + regex)
- Phone number (Indian 10-digit)
- Email address
- Course name (knowledge base match)
- Date / time preference
- City (Indian cities list)
"""

import re
import json
import logging
from pathlib import Path

import nltk
from nltk import pos_tag, ne_chunk, word_tokenize
from nltk.tree import Tree

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"

# ── Static lists ────────────────────────────────────────────────────────────

INDIAN_CITIES = [
    "bhopal", "indore", "jabalpur", "gwalior", "ujjain", "sagar",
    "ratlam", "satna", "dewas", "murwara", "chhindwara", "rewa",
    "singrauli", "burhanpur", "khandwa", "bhind", "shivpuri", "vidisha",
    "delhi", "mumbai", "kolkata", "chennai", "bangalore", "hyderabad",
    "pune", "ahmedabad", "jaipur", "lucknow", "kanpur", "nagpur",
    "patna", "bhopal", "chandigarh", "coimbatore", "agra", "varanasi",
    "surat", "vadodara", "nashik", "rajkot", "meerut", "faridabad"
]

RELATIVE_DATE_KEYWORDS = {
    "tomorrow": "tomorrow",
    "today": "today",
    "day after tomorrow": "day after tomorrow",
    "next week": "next week",
    "monday": "Monday",
    "tuesday": "Tuesday",
    "wednesday": "Wednesday",
    "thursday": "Thursday",
    "friday": "Friday",
    "saturday": "Saturday",
}

TIME_KEYWORDS = {
    "morning": "morning (9 AM – 12 PM)",
    "afternoon": "afternoon (12 PM – 3 PM)",
    "evening": "evening (3 PM – 5 PM)",
    "9am": "9:00 AM",
    "10am": "10:00 AM",
    "11am": "11:00 AM",
    "12pm": "12:00 PM",
    "1pm": "1:00 PM",
    "2pm": "2:00 PM",
    "3pm": "3:00 PM",
    "4pm": "4:00 PM",
}


class EntityExtractor:
    """
    Extracts structured entities from raw user text using a combination of
    NLTK NE chunking, regular expressions, and knowledge-base lookups.
    """

    def __init__(self):
        """Load knowledge base and build course list for matching."""
        self.courses = self._load_courses()
        logger.info("EntityExtractor initialized with %d known courses.", len(self.courses))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, text: str) -> dict:
        """
        Extract all entities from user input.

        Args:
            text: Raw user message.

        Returns:
            Dict with keys: person_name, phone, email, course, date_preference,
            time_preference, city. Each value is None if not found.
        """
        entities = {
            "person_name": self._extract_person(text),
            "phone": self._extract_phone(text),
            "email": self._extract_email(text),
            "course": self._extract_course(text),
            "date_preference": self._extract_date(text),
            "time_preference": self._extract_time(text),
            "city": self._extract_city(text),
        }
        logger.debug("Extracted entities: %s", entities)
        return entities

    # ------------------------------------------------------------------
    # Individual extractors
    # ------------------------------------------------------------------

    def _extract_person(self, text: str) -> str | None:
        """
        Extract person name using NLTK NE chunking and regex fallback.

        Args:
            text: Raw user input.

        Returns:
            Person name string or None.
        """
        # Common non-name words to reject
        NON_NAME_WORDS = {
            "interested", "looking", "applying", "here", "student",
            "not", "from", "also", "just", "want", "trying", "about",
        }

        # Regex fallback first (fast, handles "my name is X")
        # Capture at most 3 space-separated words (real names are short)
        patterns = [
            r"my name is ([A-Za-z]+(?: [A-Za-z]+){0,2})",
            r"i'm ([A-Za-z]+(?: [A-Za-z]+){0,1})(?:\s|$|,)",
            r"name:?\s*([A-Za-z]+(?: [A-Za-z]+){0,2})",
            r"call me ([A-Za-z]+(?: [A-Za-z]+){0,1})",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                name = m.group(1).strip()
                first_word = name.split()[0].lower()
                # Basic sanity check (at least 2 chars, not a common word)
                if len(name) >= 2 and first_word not in NON_NAME_WORDS:
                    return name

        # NLTK NE chunk
        try:
            tokens = word_tokenize(text)
            pos_tags = pos_tag(tokens)
            tree = ne_chunk(pos_tags, binary=False)
            for subtree in tree:
                if isinstance(subtree, Tree) and subtree.label() == "PERSON":
                    name = " ".join(word for word, _ in subtree.leaves())
                    if len(name) >= 2:
                        return name
        except Exception as exc:
            logger.warning("NE chunking error: %s", exc)

        return None

    def _extract_phone(self, text: str) -> str | None:
        """
        Extract Indian 10-digit mobile number.

        Args:
            text: Raw user input.

        Returns:
            Phone number string or None.
        """
        match = re.search(r"\b[6-9]\d{9}\b", text)
        return match.group(0) if match else None

    def _extract_email(self, text: str) -> str | None:
        """
        Extract email address.

        Args:
            text: Raw user input.

        Returns:
            Email string or None.
        """
        match = re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text)
        return match.group(0).lower() if match else None

    def _extract_course(self, text: str) -> str | None:
        """
        Match course names from the knowledge base against user input.

        Args:
            text: Raw user input.

        Returns:
            Matched course short name or None.
        """
        text_lower = text.lower()
        # Sort by length descending to match longer names first
        for course in sorted(self.courses, key=len, reverse=True):
            if course.lower() in text_lower:
                return course
        return None

    def _extract_date(self, text: str) -> str | None:
        """
        Extract a date/day preference from user input.

        Args:
            text: Raw user input.

        Returns:
            Normalised date/day string or None.
        """
        text_lower = text.lower()
        for keyword, normalised in RELATIVE_DATE_KEYWORDS.items():
            if keyword in text_lower:
                return normalised

        # DD/MM/YYYY or YYYY-MM-DD
        m = re.search(r"\b(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4})\b", text)
        if m:
            return m.group(1)

        return None

    def _extract_time(self, text: str) -> str | None:
        """
        Extract time preference from user input.

        Args:
            text: Raw user input.

        Returns:
            Normalised time string or None.
        """
        text_lower = text.lower().replace(" ", "").replace(":", "")
        for keyword, normalised in TIME_KEYWORDS.items():
            if keyword in text_lower:
                return normalised

        # HH:MM AM/PM pattern
        m = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", text, re.IGNORECASE)
        if m:
            return m.group(1)

        return None

    def _extract_city(self, text: str) -> str | None:
        """
        Detect city name from a predefined list of Indian cities.

        Args:
            text: Raw user input.

        Returns:
            City name string or None.
        """
        text_lower = text.lower()
        for city in INDIAN_CITIES:
            if city in text_lower:
                return city.capitalize()
        return None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_courses(self) -> list[str]:
        """Load course names (full and short) from knowledge_base.json."""
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                kb = json.load(f)
            courses: list[str] = []
            for level in ("ug", "pg"):
                for course in kb["courses"][level]:
                    courses.append(course["short"])
                    courses.append(course["name"])
            return courses
        except Exception as exc:
            logger.error("Failed to load knowledge base for courses: %s", exc)
            return []
