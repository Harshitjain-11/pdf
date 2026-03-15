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
    user_message: str = (data.get("message") or "").strip()
    session_id: str = (data.get("session_id") or "default").strip()

    if not user_message:
        return jsonify({"error": "Empty message."}), 400

    logger.info("session=%s message=%r", session_id, user_message)

    # Context resolution (pronoun / anaphora)
    resolved_message = context_manager.resolve_text(session_id, user_message)

    # Entity extraction
    entities = entity_extractor.extract(resolved_message)

    # Intent classification
    classification = intent_classifier.classify(resolved_message, session_id=session_id)
    intent: str = classification["intent"]
    confidence: float = classification["confidence"]

    # Slot-filling state machine
    booking_id: str | None = None
    slot_prompt: str | None = None
    current_slot_state = slot_manager.get_state(session_id)

    if intent == "slot_booking" and not slot_manager.is_active(session_id):
        # Start booking flow
        slot_prompt = slot_manager.start_booking(session_id, entities)
        current_slot_state = slot_manager.get_state(session_id)
    elif slot_manager.is_active(session_id):
        # Continue booking flow
        slot_prompt, completed = slot_manager.process_input(session_id, user_message, entities)
        current_slot_state = slot_manager.get_state(session_id)

        if completed:
            slots = slot_manager.get_slots(session_id)
            booking_id = save_appointment(slots)
            save_lead(session_id, {**entities, **slots})
        elif current_slot_state == SlotState.CANCELLED:
            # Save partial lead on cancellation
            slots = slot_manager.get_slots(session_id)
            save_lead(session_id, {**entities, **slots})

    # Generate response
    ctx = context_manager.get_or_create(session_id)
    reply, quick_replies = response_generator.generate(
        intent=intent,
        entities=entities,
        context=ctx,
        slot_state=current_slot_state.value,
        extra_data={"slot_prompt": slot_prompt},
    )

    # Slot prompt overrides generated response during active/completed/cancelled slot flow
    if slot_prompt:
        reply = slot_prompt
        quick_replies = QUICK_REPLIES.get("slot_booking", []) if current_slot_state in (
            SlotState.COMPLETED, SlotState.CANCELLED
        ) else []

    # Update context
    context_manager.update(
        session_id=session_id,
        user_message=user_message,
        bot_response=reply,
        intent=intent,
        entities=entities,
        slot_state=current_slot_state.value,
    )

    # Persist enquiry log
    try:
        save_enquiry_log(session_id, user_message, reply, intent, confidence)
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
