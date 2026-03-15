"""
Flask application entry point for the College Enquiry Chatbot.

Routes:
  GET  /            → Chat UI
  POST /chat        → Main chatbot endpoint
  POST /slot/cancel → Cancel active slot booking
  GET  /health      → Health check
  GET  /admin       → Admin panel (HTTP Basic Auth)
"""

import os
import io
import re
import csv
import json
import logging
import secrets
from functools import wraps
from pathlib import Path

from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    Response,
)
from flask_cors import CORS

# Internal modules
from chatbot.nlp_engine import NLPEngine
from chatbot.intent_classifier import IntentClassifier
from chatbot.entity_extractor import EntityExtractor
from chatbot.slot_manager import SlotManager, SlotState
from chatbot.context_manager import ContextManager
from chatbot.response_generator import ResponseGenerator, QUICK_REPLIES
from database.db_manager import (
    init_db,
    save_appointment,
    save_enquiry_log,
    save_lead,
    get_appointment_by_id,
    get_all_appointments,
    get_all_leads,
    update_appointment_status,
    get_stats,
)

# ── Logging setup ─────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# ── Quick-reply button text → intent (BUG 1 FIX) ────────────────────────────

QUICK_REPLY_MAP: dict[str, str] = {
    "Book Appointment": "slot_booking",
    "Courses Offered": "courses_offered",
    "Fee Structure": "fees_structure",
    "Admission Process": "admission_process",
    "Contact Us": "contact_info",
    "Scholarship Info": "scholarship",
    "Eligibility Criteria": "eligibility",
    "Eligibility": "eligibility",
    "Documents Needed": "documents_needed",
    "Last Date to Apply": "last_date",
    "Last Date": "last_date",
    "B.Tech Details": "course_detail",
    "Apply Now": "slot_booking",
    "Book Campus Visit": "slot_booking",
    "Cancel Booking": "cancel_booking",
    "Placement Info": "placement_info",
    "Hostel Info": "hostel_info",
    "Faculty Info": "faculty_info",
    "Campus Life": "campus_life",
}

# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app)

# ── NLP / chatbot singletons (initialized at startup) ─────────────────────────

nlp_engine: NLPEngine | None = None
intent_classifier: IntentClassifier | None = None
entity_extractor: EntityExtractor | None = None
slot_manager: SlotManager | None = None
context_manager: ContextManager | None = None
response_generator: ResponseGenerator | None = None
model_loaded: bool = False


def _initialize_components() -> None:
    """Initialize all chatbot components and the database."""
    global nlp_engine, intent_classifier, entity_extractor
    global slot_manager, context_manager, response_generator, model_loaded

    logger.info("Initializing database…")
    init_db()

    logger.info("Loading NLP components…")
    nlp_engine = NLPEngine()
    intent_classifier = IntentClassifier(nlp_engine)
    entity_extractor = EntityExtractor()
    slot_manager = SlotManager()
    context_manager = ContextManager()
    response_generator = ResponseGenerator()

    model_loaded = True
    logger.info("All components initialized successfully.")


# ── Message preprocessing (PART H edge cases) ───────────────────────────────

# Emoji pattern for stripping
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\U00002600-\U000026FF"
    "]+",
    flags=re.UNICODE,
)


def _preprocess_message(text: str) -> str:
    """
    Clean user input: lowercase, strip emoji, collapse whitespace,
    remove excess dots/special chars.
    """
    text = _EMOJI_RE.sub("", text)
    text = re.sub(r"\.{2,}", " ", text)
    text = re.sub(r"\?{2,}", "?", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── HTTP Basic Auth helper ────────────────────────────────────────────────────

ADMIN_USERNAME = os.environ.get("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "sunrise2025")


def _check_auth(username: str, password: str) -> bool:
    return secrets.compare_digest(username, ADMIN_USERNAME) and secrets.compare_digest(
        password, ADMIN_PASSWORD
    )


def _require_auth(f):
    """Decorator that enforces HTTP Basic Auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not _check_auth(auth.username, auth.password):
            return Response(
                "Authentication required.",
                401,
                {"WWW-Authenticate": 'Basic realm="Admin"'},
            )
        return f(*args, **kwargs)
    return decorated


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/")
def index():
    """Render the chat UI."""
    return render_template("index.html")


@app.post("/chat")
def chat():
    """
    Main chatbot endpoint.

    Request JSON: {message: str, session_id: str}
    Response JSON: {
        reply: str,
        quick_replies: list[str],
        intent: str,
        confidence: float,
        slot_state: str,
        booking_id: str | null
    }
    """
    if not model_loaded:
        return jsonify({"error": "Model not loaded yet. Please retry."}), 503

    data = request.get_json(silent=True) or {}
    raw_message: str = (data.get("message") or "").strip()
    session_id: str = (data.get("session_id") or "default").strip()

    if not raw_message:
        return jsonify({"error": "Empty message."}), 400

    logger.info("session=%s message=%r", session_id, raw_message)

    # ── BUG 1 FIX: Quick-reply button → skip NLP ──────────────────────
    if raw_message in QUICK_REPLY_MAP:
        intent = QUICK_REPLY_MAP[raw_message]
        confidence = 1.0
        entities: dict = {}

        # Handle cancel from quick reply
        if intent == "cancel_booking" and slot_manager.is_active(session_id):
            slot_manager.cancel(session_id)
            return jsonify({
                "reply": "Theek hai, booking cancel kar di! 😊 Koi aur cheez mein help karoon?",
                "quick_replies": ["Courses Offered", "Fee Structure", "Admission Process", "Contact Us"],
                "intent": "cancel_booking",
                "confidence": 1.0,
                "slot_state": "CANCELLED",
                "booking_id": None,
            })

        # course_detail → treat as courses_offered
        if intent == "course_detail":
            intent = "courses_offered"
    else:
        # ── Preprocessing (PART H edge cases) ─────────────────────────
        user_message = _preprocess_message(raw_message)

        # Context resolution (pronoun / anaphora)
        resolved_message = context_manager.resolve_text(session_id, user_message)

        # Entity extraction
        entities = entity_extractor.extract(resolved_message)

        # ── BUG 6: Compound query check ───────────────────────────────
        compound = intent_classifier.classify_compound(resolved_message)
        if compound and not slot_manager.is_active(session_id):
            ctx = context_manager.get_or_create(session_id)
            combined_response = ""
            for comp_intent, comp_conf in compound:
                r, _ = response_generator.generate(
                    intent=comp_intent,
                    entities=entities,
                    context=ctx,
                )
                combined_response += r + "\n\n─────────\n\n"

            quick_replies = QUICK_REPLIES.get(compound[0][0], QUICK_REPLIES.get("fallback", []))
            context_manager.update(
                session_id=session_id,
                user_message=raw_message,
                bot_response=combined_response.strip(),
                intent="compound",
                entities=entities,
            )
            try:
                save_enquiry_log(session_id, raw_message, combined_response.strip(), "compound", 1.0)
            except Exception as exc:
                logger.error("Failed to save enquiry log: %s", exc)
            return jsonify({
                "reply": combined_response.strip(),
                "quick_replies": quick_replies,
                "intent": "compound",
                "confidence": 1.0,
                "slot_state": "IDLE",
                "booking_id": None,
            })

        # Intent classification
        classification = intent_classifier.classify(resolved_message, session_id=session_id)
        intent = classification["intent"]
        confidence = classification["confidence"]

    # ── Slot-filling state machine ────────────────────────────────────
    booking_id: str | None = None
    slot_prompt: str | None = None
    current_slot_state = slot_manager.get_state(session_id)

    if intent == "slot_booking" and not slot_manager.is_active(session_id):
        slot_prompt = slot_manager.start_booking(session_id, entities)
        current_slot_state = slot_manager.get_state(session_id)
    elif slot_manager.is_active(session_id):
        slot_prompt, completed = slot_manager.process_input(
            session_id,
            raw_message,
            entities,
            detected_intent=intent,
            response_generator=response_generator,
            context=context_manager.get_or_create(session_id),
        )
        current_slot_state = slot_manager.get_state(session_id)

        if completed:
            slots = slot_manager.get_slots(session_id)
            booking_id = save_appointment(slots)
            save_lead(session_id, {**entities, **slots})
        elif current_slot_state == SlotState.CANCELLED:
            slots = slot_manager.get_slots(session_id)
            save_lead(session_id, {**entities, **slots})

    # ── Generate response ─────────────────────────────────────────────
    ctx = context_manager.get_or_create(session_id)
    reply, quick_replies = response_generator.generate(
        intent=intent,
        entities=entities,
        context=ctx,
        slot_state=current_slot_state.value,
        extra_data={"slot_prompt": slot_prompt},
    )

    # Slot prompt overrides generated response
    if slot_prompt:
        reply = slot_prompt
        if current_slot_state in (SlotState.COMPLETED, SlotState.CANCELLED):
            quick_replies = ["Courses Offered", "Fee Structure", "Admission Process", "Contact Us"]
        else:
            quick_replies = ["Cancel Booking"]

    # ── PART H: Repeat question detection ─────────────────────────────
    history = context_manager.get_history(session_id)
    if len(history) >= 1 and not slot_prompt:
        prev_intents = [h.get("intent") for h in history[-3:]]
        if intent in prev_intents and intent not in ("greeting", "goodbye", "fallback", "slot_booking"):
            reply = (
                "(Yeh information pehle bhi share ki thi, par dobara yahan hai!)\n\n"
                + reply
            )

    # ── Update context ────────────────────────────────────────────────
    context_manager.update(
        session_id=session_id,
        user_message=raw_message,
        bot_response=reply,
        intent=intent,
        entities=entities,
        slot_state=current_slot_state.value,
    )

    # Persist enquiry log
    try:
        save_enquiry_log(session_id, raw_message, reply, intent, confidence)
    except Exception as exc:
        logger.error("Failed to save enquiry log: %s", exc)

    return jsonify(
        {
            "reply": reply,
            "quick_replies": quick_replies,
            "intent": intent,
            "confidence": confidence,
            "slot_state": current_slot_state.value,
            "booking_id": booking_id,
        }
    )


@app.post("/slot/cancel")
def cancel_slot():
    """Cancel an active slot booking for the given session."""
    data = request.get_json(silent=True) or {}
    session_id: str = (data.get("session_id") or "default").strip()
    slot_manager.cancel(session_id)
    return jsonify({"status": "cancelled", "session_id": session_id})


@app.get("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "model_loaded": model_loaded})


# ── Admin panel ───────────────────────────────────────────────────────────────


@app.get("/admin")
@_require_auth
def admin():
    """Render the admin panel."""
    stats = get_stats()
    appointments = get_all_appointments()
    leads = get_all_leads()
    return render_template(
        "admin.html",
        stats=stats,
        appointments=appointments,
        leads=leads,
    )


@app.post("/admin/appointment/<booking_id>/confirm")
@_require_auth
def admin_confirm_appointment(booking_id: str):
    """Mark an appointment as confirmed."""
    updated = update_appointment_status(booking_id, "confirmed")
    return jsonify({"success": updated, "booking_id": booking_id})


@app.post("/admin/appointment/<booking_id>/cancel")
@_require_auth
def admin_cancel_appointment(booking_id: str):
    """Mark an appointment as cancelled."""
    updated = update_appointment_status(booking_id, "cancelled")
    return jsonify({"success": updated, "booking_id": booking_id})


@app.get("/admin/export/appointments")
@_require_auth
def export_appointments():
    """Export all appointments as a CSV file."""
    appointments = get_all_appointments()
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "id", "booking_id", "name", "phone", "email",
            "course_interest", "preferred_date", "preferred_time",
            "status", "created_at"
        ]
    )
    writer.writeheader()
    writer.writerows(appointments)
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=appointments.csv"},
    )


@app.get("/admin/stats")
@_require_auth
def admin_stats():
    """Return JSON statistics for the admin panel charts."""
    return jsonify(get_stats())


# ── Error handlers ────────────────────────────────────────────────────────────


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found."}), 404


@app.errorhandler(500)
def server_error(e):
    logger.exception("Internal server error: %s", e)
    return jsonify({"error": "Internal server error. Please try again."}), 500


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _initialize_components()
    app.run(debug=False, host="0.0.0.0", port=5000)
