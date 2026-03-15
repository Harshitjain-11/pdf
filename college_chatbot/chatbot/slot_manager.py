"""
Slot Manager for the College Enquiry Chatbot.

Implements a finite-state machine for collecting booking information
through multi-turn conversation.
"""

import re
import uuid
import logging
from datetime import datetime, date
from enum import Enum

logger = logging.getLogger(__name__)


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
    SlotState.COLLECTING_NAME: (
        "Great! I'll help you book a counselling appointment. "
        "Could you please tell me your **full name**?"
    ),
    SlotState.COLLECTING_PHONE: "Thank you, {name}! What is your **10-digit mobile number**?",
    SlotState.COLLECTING_EMAIL: (
        "Got it! What is your **email address**? "
        "(You can type 'skip' to skip this step)"
    ),
    SlotState.COLLECTING_COURSE: (
        "Which **course** are you interested in? "
        "(e.g., B.Tech CS, BCA, MBA, MCA)"
    ),
    SlotState.COLLECTING_DATE: (
        "What is your **preferred date** for the visit? "
        "(Monday–Saturday, format: DD/MM/YYYY or e.g. 'Monday', 'next week')"
    ),
    SlotState.COLLECTING_TIME: (
        "What is your **preferred time**? "
        "Office hours: 9 AM – 5 PM. (e.g., '10am', 'morning', '2pm')"
    ),
    SlotState.CONFIRMING: (
        "✅ Please confirm your booking details:\n"
        "👤 Name: {name}\n"
        "📞 Phone: {phone}\n"
        "📧 Email: {email}\n"
        "📚 Course: {course_interest}\n"
        "📅 Date: {preferred_date}\n"
        "⏰ Time: {preferred_time}\n\n"
        "Type **'haan'** or **'yes'** to confirm, or **'nahi'**/**'no'** to cancel."
    ),
}

CANCEL_WORDS = {"cancel", "stop", "nahi chahiye", "quit", "exit", "nahi", "no thanks", "nevermind"}
VALID_HOURS = range(9, 17)  # 9 AM – 4 PM (last slot at 4 PM)


class SlotManager:
    """
    State machine for collecting counselling appointment details.

    Holds per-session state in a dict. Persisting/loading is handled
    externally (in ContextManager / Flask session).
    """

    def __init__(self):
        """Initialize slot manager with in-memory session store."""
        # session_id -> slot_dict
        self._sessions: dict[str, dict] = {}

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
        Begin the booking flow, pre-filling any already-known entities.

        Args:
            session_id: Current session ID.
            entities: Entities extracted from the initial trigger message.

        Returns:
            Next prompt to display to the user.
        """
        slots = {
            "state": SlotState.IDLE,
            "name": entities.get("person_name"),
            "phone": entities.get("phone"),
            "email": entities.get("email"),
            "_email_collected": entities.get("email") is not None,  # track if email step is done
            "course_interest": entities.get("course"),
            "preferred_date": entities.get("date_preference"),
            "preferred_time": entities.get("time_preference"),
            "booking_id": None,
            "status": "in_progress",
            "created_at": datetime.utcnow().isoformat(),
        }
        self._sessions[session_id] = slots
        return self._advance(session_id)

    def process_input(self, session_id: str, text: str, entities: dict) -> tuple[str, bool]:
        """
        Process user input during slot filling.

        Args:
            session_id: Current session ID.
            text: Raw user input.
            entities: Freshly extracted entities from this turn.

        Returns:
            Tuple of (response_text, is_completed).
        """
        if session_id not in self._sessions:
            return "No active booking found. Type 'book appointment' to start.", False

        slots = self._sessions[session_id]
        state = SlotState(slots["state"])

        # Check for cancellation
        if self._is_cancel(text):
            slots["state"] = SlotState.CANCELLED
            slots["status"] = "cancelled"
            return (
                "❌ Booking cancelled. If you change your mind, just say "
                "'book appointment' again. How else can I help you?",
                False,
            )

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
        # Validate: at least 2 characters and looks like a name
        if len(name) < 2 or re.match(r"^\d+$", name):
            return "Please enter your **full name** (at least first and last name)."
        slots["name"] = name
        return self._advance_slots(slots)

    def _collect_phone(self, slots: dict, text: str, entities: dict) -> str:
        phone = entities.get("phone") or re.search(r"\b[6-9]\d{9}\b", text)
        if isinstance(phone, re.Match):
            phone = phone.group(0)
        if not phone or not re.fullmatch(r"[6-9]\d{9}", str(phone)):
            return "Please enter a valid **10-digit Indian mobile number** (starting with 6-9)."
        slots["phone"] = str(phone)
        return self._advance_slots(slots)

    def _collect_email(self, slots: dict, text: str, entities: dict) -> str:
        if text.strip().lower() in {"skip", "no", "nahi", "n/a", "-"}:
            slots["email"] = None
            slots["_email_collected"] = True
            return self._advance_slots(slots)
        email = entities.get("email")
        if not email:
            m = re.search(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b", text)
            email = m.group(0) if m else None
        if not email:
            return "Please enter a valid **email address** (or type 'skip' to skip)."
        slots["email"] = email.lower()
        slots["_email_collected"] = True
        return self._advance_slots(slots)

    def _collect_course(self, slots: dict, text: str, entities: dict) -> str:
        course = entities.get("course") or text.strip()
        if len(course) < 2:
            return "Please specify the **course** you're interested in (e.g., B.Tech CS, MBA)."
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
        if text_lower in {"haan", "yes", "y", "confirm", "ok", "okay", "ha"}:
            booking_id = f"ITM{uuid.uuid4().hex[:8].upper()}"
            slots["booking_id"] = booking_id
            slots["state"] = SlotState.COMPLETED
            slots["status"] = "pending"
            return (
                f"🎉 **Booking Confirmed!**\n"
                f"Your booking ID is: **{booking_id}**\n"
                f"Our team will contact you at {slots.get('phone')} to confirm your appointment.\n"
                f"You can also call us at +91-751-2970300. See you at ITM Gwalior! 😊",
                True,
            )
        elif text_lower in {"nahi", "no", "n", "cancel", "nope"}:
            slots["state"] = SlotState.CANCELLED
            slots["status"] = "cancelled"
            return (
                "❌ Booking cancelled. If you change your mind, just say "
                "'book appointment' again. How else can I help you?",
                False,
            )
        else:
            return (
                "Please type **'yes'**/**'haan'** to confirm or **'no'**/**'nahi'** to cancel.",
                False,
            )

    # ------------------------------------------------------------------
    # State machine helpers
    # ------------------------------------------------------------------

    def _advance(self, session_id: str) -> str:
        """Determine the next state and return its prompt."""
        slots = self._sessions[session_id]
        return self._advance_slots(slots)

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

        # All slots filled → confirm
        slots["state"] = SlotState.CONFIRMING
        email_display = slots.get("email") or "Not provided"
        return STATE_PROMPTS[SlotState.CONFIRMING].format(
            name=slots["name"],
            phone=slots["phone"],
            email=email_display,
            course_interest=slots["course_interest"],
            preferred_date=slots["preferred_date"],
            preferred_time=slots["preferred_time"],
        )

    # ------------------------------------------------------------------
    # Validators
    # ------------------------------------------------------------------

    def _validate_date(self, date_str: str) -> str | None:
        """
        Validate that the date is not in the past and not a Sunday.

        Args:
            date_str: Date string from user.

        Returns:
            Error message string if invalid, else None.
        """
        date_str_lower = date_str.lower().strip()

        # Known relative day/period keywords
        relative_ok = {
            "tomorrow", "today", "next week", "monday", "tuesday",
            "wednesday", "thursday", "friday", "saturday",
            "day after tomorrow"
        }
        if date_str_lower in relative_ok:
            if date_str_lower == "sunday":
                return "❌ The college is closed on Sundays. Please choose Monday–Saturday."
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
                    return "❌ The date cannot be in the past. Please choose a future date."
                if chosen.weekday() == 6:  # Sunday
                    return "❌ The college is closed on Sundays. Please choose Monday–Saturday."
                return None
            except ValueError:
                return "❌ Invalid date format. Please use DD/MM/YYYY (e.g., 15/07/2025)."

        # Accept "15th July" style free-text (best-effort, at least 5 chars with a digit)
        if len(date_str) >= 5 and re.search(r"\d", date_str):
            return None

        # Accept day names as substrings ("next monday", "this friday")
        day_names = {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday"}
        if any(d in date_str_lower for d in day_names):
            return None

        return "❌ Please enter a valid date (e.g., '15/07/2025', 'Monday', 'tomorrow')."

    def _validate_time(self, time_str: str) -> str | None:
        """
        Validate that the time is within office hours (9 AM – 4 PM).

        Args:
            time_str: Time string from user.

        Returns:
            Error message string if invalid, else None.
        """
        time_lower = time_str.lower().strip().replace(" ", "")

        # Keyword-based
        if "morning" in time_lower:
            return None
        if "afternoon" in time_lower:
            return None
        if "evening" in time_lower:
            return "❌ Evening slots are not available. Office hours: 9 AM – 5 PM (last slot 4 PM)."

        # Must contain a digit to be a valid time entry
        if not re.search(r"\d", time_lower):
            return "❌ Please enter a valid time (e.g., '10am', 'morning', '2pm')."

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
                return "❌ Please choose a time between 9 AM and 4 PM (office hours)."
            return None

        return "❌ Please enter a valid time (e.g., '10am', 'morning', '2pm')."

    def _is_cancel(self, text: str) -> bool:
        """Return True if the user wants to cancel the booking."""
        text_lower = text.lower().strip()
        return any(cancel_word in text_lower for cancel_word in CANCEL_WORDS)
