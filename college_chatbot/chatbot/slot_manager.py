"""
Slot Manager for the College Enquiry Chatbot.

Implements a finite-state machine for collecting booking information
through multi-turn conversation with interrupt detection and validation.
"""

import re
import uuid
import json
import logging
from datetime import datetime, date, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"


class SlotState(str, Enum):
    """States in the slot-filling state machine."""
    IDLE = "IDLE"
    COLLECTING_NAME = "COLLECTING_NAME"
    COLLECTING_PHONE = "COLLECTING_PHONE"
    COLLECTING_EMAIL = "COLLECTING_EMAIL"
    COLLECTING_COURSE = "COLLECTING_COURSE"
    COLLECTING_DATE = "COLLECTING_DATE"
    COLLECTING_TIME = "COLLECTING_TIME"
    CONFIRMING = "CONFIRMING"
    COMPLETED = "COMPLETED"
    CANCELLED = "CANCELLED"


# Questions to ask at each state
STATE_PROMPTS = {
    SlotState.COLLECTING_NAME: "Zaroor! Pehle aapka poora naam batayein? 😊",
    SlotState.COLLECTING_PHONE: "Shukriya {name} ji! 😊 Aapka 10-digit mobile number kya hai?",
    SlotState.COLLECTING_EMAIL: (
        "📧 Aapka email address? (optional — 'skip' type karke skip kar sakte ho)"
    ),
    SlotState.COLLECTING_COURSE: (
        "🎓 Kaunse course mein interest hai? (e.g., B.Tech CS, BCA, MBA, MCA)"
    ),
    SlotState.COLLECTING_DATE: (
        "📅 Kab aana chahte ho? (Mon-Sat, future date — e.g., 'kal', 'Monday', '15/07/2025')"
    ),
    SlotState.COLLECTING_TIME: (
        "🕐 Kaun sa time? Office hours: 9 AM – 4 PM\n"
        "(e.g., '10am', 'morning', 'afternoon', '2pm')"
    ),
}

CANCEL_WORDS = [
    "cancel", "stop", "band karo", "nahi chahiye",
    "exit", "quit", "wapas", "chodo", "rehne do",
    "no booking", "don't book", "mat karo"
]

INTERRUPTIBLE_INTENTS = [
    "greeting", "fees_structure", "courses_offered",
    "admission_process", "eligibility", "documents_needed",
    "last_date", "scholarship", "hostel_info", "contact_info",
    "exam_schedule", "result_status", "placement_info",
    "campus_life", "faculty_info", "anti_ragging",
    "lateral_entry", "refund_policy", "naac_info",
    "fallback", "angry_user", "out_of_scope"
]

# Names that should not be accepted (button labels and common non-names)
INVALID_NAMES = {
    "book", "appointment", "apply", "courses",
    "fee", "structure", "admission", "process",
    "contact", "scholarship", "hi", "hello",
    "cancel", "booking", "campus", "visit",
    "yes", "no", "haan", "nahi", "ok", "okay"
}

# Multi-word button labels that should be rejected as names
BUTTON_LABELS = {
    "book appointment", "apply now", "book campus visit",
    "cancel booking", "courses offered", "fee structure",
    "admission process", "contact us", "scholarship info",
    "eligibility criteria", "documents needed", "last date to apply",
    "b.tech details", "placement info", "hostel info",
    "faculty info", "campus life"
}

VALID_HOURS = range(9, 17)  # 9 AM – 4 PM (last slot at 4 PM)


def _load_course_names() -> list[str]:
    """Load course names from knowledge base for matching."""
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb = json.load(f)
        courses = []
        for level in ("ug", "pg"):
            for course in kb["courses"][level]:
                courses.append(course["short"])
                courses.append(course["name"])
        return courses
    except Exception:
        return []


class SlotManager:
    """
    State machine for collecting counselling appointment details.

    Holds per-session state in a dict with interrupt detection,
    cancel support, and input validation with retry limits.
    """

    def __init__(self):
        """Initialize slot manager with in-memory session store."""
        self._sessions: dict[str, dict] = {}
        self._courses = _load_course_names()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_state(self, session_id: str) -> SlotState:
        """Return current slot state for the session."""
        return SlotState(self._sessions.get(session_id, {}).get("state", SlotState.IDLE))

    def get_slots(self, session_id: str) -> dict:
        """Return current slot dict for the session."""
        return dict(self._sessions.get(session_id, {}))

    def start_booking(self, session_id: str, entities: dict) -> str:
        """
        Begin the booking flow. Always start from COLLECTING_NAME.
        Pre-fill name only if valid (not a button label).

        Args:
            session_id: Current session ID.
            entities: Entities extracted from the initial trigger message.

        Returns:
            Next prompt to display to the user.
        """
        slots = {
            "state": SlotState.COLLECTING_NAME,
            "name": None,
            "phone": None,
            "email": None,
            "_email_collected": False,
            "course_interest": None,
            "preferred_date": None,
            "preferred_time": None,
            "booking_id": None,
            "status": "in_progress",
            "created_at": datetime.utcnow().isoformat(),
            "_phone_attempts": 0,
        }
        self._sessions[session_id] = slots

        # Check if a valid name was provided in the message
        if entities.get("person_name"):
            name = entities["person_name"]
            if self._is_valid_name(name):
                slots["name"] = name
                slots["state"] = SlotState.COLLECTING_PHONE
                return STATE_PROMPTS[SlotState.COLLECTING_PHONE].format(name=name)

        return STATE_PROMPTS[SlotState.COLLECTING_NAME]

    def process_input(
        self,
        session_id: str,
        text: str,
        entities: dict,
        detected_intent: str | None = None,
        response_generator=None,
        context: dict | None = None,
    ) -> tuple[str, bool]:
        """
        Process user input during slot filling.

        Includes cancel detection and interrupt handling for non-slot questions.

        Args:
            session_id: Current session ID.
            text: Raw user input.
            entities: Freshly extracted entities from this turn.
            detected_intent: Intent detected by classifier (for interrupt detection).
            response_generator: ResponseGenerator instance (for interrupt answers).
            context: Session context dict (for interrupt answers).

        Returns:
            Tuple of (response_text, is_completed).
        """
        if session_id not in self._sessions:
            return "No active booking found. Type 'book appointment' to start.", False

        slots = self._sessions[session_id]
        state = SlotState(slots["state"])

        # STEP 1: Cancel detection (always checked first)
        if self._is_cancel(text):
            slots["state"] = SlotState.CANCELLED
            slots["status"] = "cancelled"
            return (
                "Theek hai, booking cancel kar di! 😊 "
                "Koi aur cheez mein help karoon?",
                False,
            )

        # STEP 2: Interrupt detection (non-slot question during booking)
        if (
            state not in (SlotState.IDLE, SlotState.COMPLETED, SlotState.CANCELLED)
            and detected_intent
            and detected_intent in INTERRUPTIBLE_INTENTS
            and response_generator is not None
        ):
            # Answer their question
            ctx = context or {}
            answer, _ = response_generator.generate(
                intent=detected_intent,
                entities=entities,
                context=ctx,
                slot_state=state.value,
            )
            # Resume booking from current state
            resume_prompt = self._get_slot_question(slots)
            return (
                f"{answer}\n\n"
                "─────────────────────────\n"
                "📋 Chalo apni visit booking bhi complete karte hain!\n\n"
                f"{resume_prompt}",
                False,
            )

        # Process based on current state
        if state == SlotState.COLLECTING_NAME:
            response = self._collect_name(slots, text, entities)

        elif state == SlotState.COLLECTING_PHONE:
            response = self._collect_phone(slots, text, entities)

        elif state == SlotState.COLLECTING_EMAIL:
            response = self._collect_email(slots, text, entities)

        elif state == SlotState.COLLECTING_COURSE:
            response = self._collect_course(slots, text, entities)

        elif state == SlotState.COLLECTING_DATE:
            response = self._collect_date(slots, text, entities)

        elif state == SlotState.COLLECTING_TIME:
            response = self._collect_time(slots, text, entities)

        elif state == SlotState.CONFIRMING:
            return self._confirm(slots, text)

        else:
            response = "Something went wrong. Let's start again — type 'book appointment'."

        return response, slots.get("state") == SlotState.COMPLETED

    def cancel(self, session_id: str) -> None:
        """Cancel an active booking for the given session."""
        if session_id in self._sessions:
            self._sessions[session_id]["state"] = SlotState.CANCELLED
            self._sessions[session_id]["status"] = "cancelled"

    def clear(self, session_id: str) -> None:
        """Remove session state entirely."""
        self._sessions.pop(session_id, None)

    def is_active(self, session_id: str) -> bool:
        """Return True if slot filling is currently active for the session."""
        state = self.get_state(session_id)
        return state not in (SlotState.IDLE, SlotState.COMPLETED, SlotState.CANCELLED)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _collect_name(self, slots: dict, text: str, entities: dict) -> str:
        name = entities.get("person_name") or text.strip()
        if not self._is_valid_name(name):
            return "Please apna **poora naam** batayein (at least 3 characters)."
        slots["name"] = name
        return self._advance_slots(slots)

    def _collect_phone(self, slots: dict, text: str, entities: dict) -> str:
        # Clean phone: remove spaces, dashes, +91, leading 0
        cleaned = re.sub(r"[\s\-\(\)]", "", text)
        if cleaned.startswith("+91"):
            cleaned = cleaned[3:]
        if cleaned.startswith("91") and len(cleaned) == 12:
            cleaned = cleaned[2:]
        if cleaned.startswith("0") and len(cleaned) == 11:
            cleaned = cleaned[1:]

        phone = entities.get("phone")
        if not phone:
            m = re.search(r"\b[6-9]\d{9}\b", cleaned)
            phone = m.group(0) if m else None

        if not phone or not re.fullmatch(r"[6-9]\d{9}", str(phone)):
            slots["_phone_attempts"] = slots.get("_phone_attempts", 0) + 1
            if slots["_phone_attempts"] >= 3:
                slots["state"] = SlotState.IDLE
                slots["status"] = "cancelled"
                return (
                    "Lagta hai number mein dikkat hai. Seedha humse contact "
                    "karein: +91-751-2970300 😊"
                )
            return (
                "❌ Please ek valid **10-digit Indian mobile number** enter karein "
                "(6-9 se start hona chahiye)."
            )

        slots["phone"] = str(phone)
        slots["_phone_attempts"] = 0
        return self._advance_slots(slots)

    def _collect_email(self, slots: dict, text: str, entities: dict) -> str:
        if text.strip().lower() in {"skip", "no", "nahi", "nahi hai", "n/a", "-"}:
            slots["email"] = None
            slots["_email_collected"] = True
            return self._advance_slots(slots)
        email = entities.get("email")
        if not email:
            m = re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text)
            email = m.group(0) if m else None
        if not email:
            return "Please ek valid **email address** enter karein (ya 'skip' type karein)."
        slots["email"] = email.lower()
        slots["_email_collected"] = True
        return self._advance_slots(slots)

    def _collect_course(self, slots: dict, text: str, entities: dict) -> str:
        course = entities.get("course")
        if not course:
            # Try partial match against known courses
            text_lower = text.lower().strip()
            for c in sorted(self._courses, key=len, reverse=True):
                if c.lower() in text_lower or text_lower in c.lower():
                    course = c
                    break
        if not course:
            if len(text.strip()) >= 2:
                course = text.strip()
            else:
                course_list = ", ".join(
                    c for c in self._courses if not c.startswith("B") or "." in c
                )[:200]
                return (
                    f"Koi match nahi mila. Available courses: {course_list}\n"
                    "Please course ka naam batayein."
                )
        slots["course_interest"] = course
        return self._advance_slots(slots)

    def _collect_date(self, slots: dict, text: str, entities: dict) -> str:
        preferred_date = entities.get("date_preference") or text.strip()
        error = self._validate_date(preferred_date)
        if error:
            return error
        slots["preferred_date"] = preferred_date
        return self._advance_slots(slots)

    def _collect_time(self, slots: dict, text: str, entities: dict) -> str:
        preferred_time = entities.get("time_preference") or text.strip()
        error = self._validate_time(preferred_time)
        if error:
            return error
        slots["preferred_time"] = preferred_time
        return self._advance_slots(slots)

    def _confirm(self, slots: dict, text: str) -> tuple[str, bool]:
        text_lower = text.strip().lower()
        if text_lower in {"haan", "yes", "y", "confirm", "ok", "okay", "ha", "ji"}:
            booking_id = f"ITM{uuid.uuid4().hex[:8].upper()}"
            slots["booking_id"] = booking_id
            slots["state"] = SlotState.COMPLETED
            slots["status"] = "pending"
            email_display = slots.get("email") or "Not provided"
            return (
                f"🎉 Booking confirmed!\n\n"
                f"📋 Booking ID: {booking_id}\n"
                f"📅 {slots.get('preferred_date')} ko {slots.get('preferred_time')} baje "
                f"ITM Gwalior admission office mein aaiye.\n\n"
                f"📍 Address: AB Road, Gwalior, MP 474001\n"
                f"📞 Helpline: +91-751-2970300\n\n"
                f"Koi aur sawaal?",
                True,
            )
        elif text_lower in {"nahi", "no", "n", "cancel", "nope"}:
            slots["state"] = SlotState.CANCELLED
            slots["status"] = "cancelled"
            return (
                "Theek hai, booking cancel kar di! 😊 "
                "Koi aur cheez mein help karoon?",
                False,
            )
        else:
            return (
                "Kya yeh sahi hai? **'Haan'** ya **'Nahi'** type karein.",
                False,
            )

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _advance_slots(self, slots: dict) -> str:
        """Find the next unfilled slot and update state accordingly."""
        if not slots.get("name"):
            slots["state"] = SlotState.COLLECTING_NAME
            return STATE_PROMPTS[SlotState.COLLECTING_NAME]
        if not slots.get("phone"):
            slots["state"] = SlotState.COLLECTING_PHONE
            return STATE_PROMPTS[SlotState.COLLECTING_PHONE].format(name=slots["name"])
        if not slots.get("_email_collected"):
            slots["state"] = SlotState.COLLECTING_EMAIL
            return STATE_PROMPTS[SlotState.COLLECTING_EMAIL]
        if not slots.get("course_interest"):
            slots["state"] = SlotState.COLLECTING_COURSE
            return STATE_PROMPTS[SlotState.COLLECTING_COURSE]
        if not slots.get("preferred_date"):
            slots["state"] = SlotState.COLLECTING_DATE
            return STATE_PROMPTS[SlotState.COLLECTING_DATE]
        if not slots.get("preferred_time"):
            slots["state"] = SlotState.COLLECTING_TIME
            return STATE_PROMPTS[SlotState.COLLECTING_TIME]

        # All slots filled → show confirmation summary
        slots["state"] = SlotState.CONFIRMING
        email_display = slots.get("email") or "Not provided"
        return (
            "✅ Booking Summary:\n"
            f"👤 Naam: {slots['name']}\n"
            f"📱 Phone: {slots['phone']}\n"
            f"📧 Email: {email_display}\n"
            f"🎓 Course: {slots['course_interest']}\n"
            f"📅 Date: {slots['preferred_date']}\n"
            f"🕐 Time: {slots['preferred_time']}\n\n"
            "Kya yeh sahi hai? (Haan / Nahi)"
        )

    def _get_slot_question(self, slots: dict) -> str:
        """Return the current slot question for resume after interrupt."""
        state = SlotState(slots["state"])
        if state == SlotState.COLLECTING_PHONE:
            return STATE_PROMPTS[state].format(name=slots.get("name", ""))
        return STATE_PROMPTS.get(state, "Aage badhein?")

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    def _is_valid_name(self, name: str) -> bool:
        """Validate a name: at least 3 chars, not a button label or digit-only."""
        if not name or len(name.strip()) < 3:
            return False
        stripped = name.strip()
        if re.fullmatch(r"\d+", stripped):
            return False
        # Check individual words against invalid names
        words = stripped.lower().split()
        if any(w in INVALID_NAMES for w in words):
            return False
        # Also check multi-word button labels
        if stripped.lower() in BUTTON_LABELS:
            return False
        return True

    def _validate_date(self, date_str: str) -> str | None:
        """
        Validate that the date is not in the past and not a Sunday.

        Supports: "kal"/"tomorrow", "parson"/"day after tomorrow",
        day names, DD/MM/YYYY format.
        """
        date_str_lower = date_str.lower().strip()

        # Handle Hindi relative dates
        if date_str_lower in ("kal", "tomorrow"):
            tomorrow = date.today() + timedelta(days=1)
            if tomorrow.weekday() == 6:
                return "College Sunday ko band rehta hai! Please koi aur din choose karein (Monday se Saturday)."
            return None

        if date_str_lower in ("parson", "parso", "day after tomorrow"):
            day_after = date.today() + timedelta(days=2)
            if day_after.weekday() == 6:
                return "College Sunday ko band rehta hai! Please koi aur din choose karein (Monday se Saturday)."
            return None

        # Sunday check
        if date_str_lower == "sunday":
            return "College Sunday ko band rehta hai! Please koi aur din choose karein (Monday se Saturday)."

        # Known relative day/period keywords
        relative_ok = {
            "today", "next week", "monday", "tuesday",
            "wednesday", "thursday", "friday", "saturday",
        }
        if date_str_lower in relative_ok:
            return None

        # Try to parse DD/MM/YYYY or YYYY-MM-DD
        m = re.fullmatch(r"(\d{1,2})[/\-](\d{1,2})[/\-](\d{2,4})", date_str)
        if m:
            try:
                day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
                if year < 100:
                    year += 2000
                chosen = date(year, month, day)
                if chosen < date.today():
                    return "Yeh date toh nikal gayi! Please future mein se koi date choose karein."
                if chosen.weekday() == 6:
                    return "College Sunday ko band rehta hai! Please koi aur din choose karein (Monday se Saturday)."
                return None
            except ValueError:
                return "❌ Invalid date format. Please DD/MM/YYYY use karein (e.g., 15/07/2025)."

        # Accept "15th July" style free-text
        if len(date_str) >= 5 and re.search(r"\d", date_str):
            return None

        # Accept day names as substrings
        day_names = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday"}
        if any(d in date_str_lower for d in day_names):
            return None

        return "❌ Please ek valid date enter karein (e.g., '15/07/2025', 'Monday', 'kal')."

    def _validate_time(self, time_str: str) -> str | None:
        """
        Validate that the time is within office hours (9 AM – 4 PM).

        Supports: "morning" → 10:00 AM, "afternoon" → 2:00 PM,
        "evening" → rejected (office closes 5PM, visits 4PM tak).
        """
        time_lower = time_str.lower().strip().replace(" ", "")

        if "morning" in time_lower:
            return None
        if "afternoon" in time_lower:
            return None
        if "evening" in time_lower or "shaam" in time_lower:
            return (
                "❌ Evening/Shaam ke slots available nahi hain. "
                "Office hours: 9 AM – 5 PM (visits 4 PM tak). "
                "Morning ya afternoon choose karein."
            )

        # Must contain a digit to be a valid time entry
        if not re.search(r"\d", time_lower):
            return "❌ Please ek valid time enter karein (e.g., '10am', 'morning', '2pm')."

        # HHam/HHpm pattern
        m = re.match(r"(\d{1,2})(?::(\d{2}))?(am|pm)?", time_lower)
        if m:
            hour = int(m.group(1))
            period = m.group(3) or ""
            if period == "pm" and hour != 12:
                hour += 12
            elif period == "am" and hour == 12:
                hour = 0
            if hour < 9 or hour >= 17:
                return "❌ Please 9 AM aur 4 PM ke beech ka time choose karein (office hours)."
            return None

        return "❌ Please ek valid time enter karein (e.g., '10am', 'morning', '2pm')."

    def _is_cancel(self, text: str) -> bool:
        """Return True if the user wants to cancel the booking."""
        text_lower = text.lower().strip()
        return any(cancel_word in text_lower for cancel_word in CANCEL_WORDS)
