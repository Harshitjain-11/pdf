"""
Database Manager for the College Enquiry Chatbot.

Provides SQLite database initialization and CRUD operations for:
- appointments
- enquiry_logs
- leads
"""

import uuid
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "database" / "college.db"


def _get_connection() -> sqlite3.Connection:
    """Create and return a new SQLite connection with row_factory set."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


@contextmanager
def _db():
    """Context manager that yields a connection and commits/rolls back."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Initialize the SQLite database by creating all required tables
    if they do not already exist.
    """
    create_appointments = """
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        booking_id TEXT UNIQUE NOT NULL,
        name TEXT NOT NULL,
        phone TEXT NOT NULL,
        email TEXT,
        course_interest TEXT,
        preferred_date TEXT,
        preferred_time TEXT,
        status TEXT DEFAULT 'pending',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_enquiry_logs = """
    CREATE TABLE IF NOT EXISTS enquiry_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        user_message TEXT,
        bot_response TEXT,
        intent TEXT,
        confidence REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    create_leads = """
    CREATE TABLE IF NOT EXISTS leads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        name TEXT,
        phone TEXT,
        email TEXT,
        course_interest TEXT,
        source TEXT DEFAULT 'chatbot',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """

    with _db() as conn:
        conn.execute(create_appointments)
        conn.execute(create_enquiry_logs)
        conn.execute(create_leads)

    logger.info("Database initialized at %s.", DB_PATH)


def save_appointment(slot_dict: dict) -> str:
    """
    Save a completed booking to the appointments table.

    Args:
        slot_dict: Slot data dict from SlotManager (must contain name and phone).

    Returns:
        The booking_id string assigned to this appointment.
    """
    booking_id = slot_dict.get("booking_id") or f"SIT{uuid.uuid4().hex[:8].upper()}"

    with _db() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO appointments
                (booking_id, name, phone, email, course_interest,
                 preferred_date, preferred_time, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                booking_id,
                slot_dict.get("name", ""),
                slot_dict.get("phone", ""),
                slot_dict.get("email"),
                slot_dict.get("course_interest"),
                slot_dict.get("preferred_date"),
                slot_dict.get("preferred_time"),
                slot_dict.get("status", "pending"),
                slot_dict.get("created_at", datetime.utcnow().isoformat()),
            ),
        )

    logger.info("Appointment saved: booking_id=%s.", booking_id)
    return booking_id


def save_enquiry_log(
    session_id: str,
    user_msg: str,
    bot_msg: str,
    intent: str,
    confidence: float,
) -> None:
    """
    Append a conversation turn to the enquiry_logs table.

    Args:
        session_id: Unique session identifier.
        user_msg: Raw user message.
        bot_msg: Generated bot response.
        intent: Classified intent tag.
        confidence: Classification confidence score.
    """
    with _db() as conn:
        conn.execute(
            """
            INSERT INTO enquiry_logs
                (session_id, user_message, bot_response, intent, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session_id, user_msg, bot_msg, intent, confidence),
        )


def save_lead(session_id: str, entities: dict) -> None:
    """
    Save a partial or complete lead to the leads table.

    Args:
        session_id: Unique session identifier.
        entities: Accumulated entity dict (name, phone, email, course).
    """
    name = entities.get("person_name") or entities.get("name")
    phone = entities.get("phone")
    email = entities.get("email")
    course = entities.get("course") or entities.get("course_interest")

    # Only save if we have at least some contact info
    if not any([name, phone, email]):
        return

    with _db() as conn:
        conn.execute(
            """
            INSERT INTO leads
                (session_id, name, phone, email, course_interest, source)
            VALUES (?, ?, ?, ?, ?, 'chatbot')
            """,
            (session_id, name, phone, email, course),
        )
    logger.info("Lead saved for session %s.", session_id)


def get_appointment_by_id(booking_id: str) -> dict | None:
    """
    Retrieve an appointment by its booking ID.

    Args:
        booking_id: The unique booking identifier.

    Returns:
        Dict with appointment fields, or None if not found.
    """
    with _db() as conn:
        row = conn.execute(
            "SELECT * FROM appointments WHERE booking_id = ?", (booking_id,)
        ).fetchone()
    return dict(row) if row else None


def get_all_appointments() -> list[dict]:
    """Return all appointments ordered by creation date descending."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM appointments ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_leads() -> list[dict]:
    """Return all leads ordered by creation date descending."""
    with _db() as conn:
        rows = conn.execute(
            "SELECT * FROM leads ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_appointment_status(booking_id: str, status: str) -> bool:
    """
    Update the status of an existing appointment.

    Args:
        booking_id: The booking identifier.
        status: New status string ('pending', 'confirmed', 'cancelled').

    Returns:
        True if the record was updated, False if not found.
    """
    with _db() as conn:
        cursor = conn.execute(
            "UPDATE appointments SET status = ? WHERE booking_id = ?",
            (status, booking_id),
        )
    updated = cursor.rowcount > 0
    if updated:
        logger.info("Appointment %s status updated to '%s'.", booking_id, status)
    return updated


def get_stats() -> dict:
    """
    Compute basic statistics for the admin panel.

    Returns:
        Dict with total_appointments, pending_count, confirmed_count,
        total_leads, today_logs, intent_distribution.
    """
    today = datetime.utcnow().date().isoformat()
    with _db() as conn:
        total_appts = conn.execute("SELECT COUNT(*) FROM appointments").fetchone()[0]
        pending = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE status='pending'"
        ).fetchone()[0]
        confirmed = conn.execute(
            "SELECT COUNT(*) FROM appointments WHERE status='confirmed'"
        ).fetchone()[0]
        total_leads = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
        today_logs = conn.execute(
            "SELECT COUNT(*) FROM enquiry_logs WHERE DATE(timestamp)=?", (today,)
        ).fetchone()[0]
        intent_rows = conn.execute(
            """
            SELECT intent, COUNT(*) as cnt
            FROM enquiry_logs
            GROUP BY intent
            ORDER BY cnt DESC
            LIMIT 10
            """
        ).fetchall()
    return {
        "total_appointments": total_appts,
        "pending_count": pending,
        "confirmed_count": confirmed,
        "total_leads": total_leads,
        "today_logs": today_logs,
        "intent_distribution": [dict(r) for r in intent_rows],
    }
