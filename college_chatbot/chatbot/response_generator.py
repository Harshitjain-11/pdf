"""
Response Generator for the College Enquiry Chatbot.

Generates responses by filling knowledge-base templates, building lists,
handling slot prompts, and returning contextually relevant quick replies.
"""

import json
import random
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
KB_PATH = BASE_DIR / "data" / "knowledge_base.json"

# Quick reply suggestions per intent
QUICK_REPLIES: dict[str, list[str]] = {
    "greeting": [
        "Admission Process",
        "Courses Offered",
        "Fee Structure",
        "Book Appointment",
    ],
    "admission_process": [
        "Documents Needed",
        "Last Date to Apply",
        "Eligibility Criteria",
        "Fee Structure",
    ],
    "courses_offered": [
        "B.Tech Details",
        "Fee Structure",
        "Eligibility",
        "Scholarship Info",
    ],
    "eligibility": [
        "Courses Offered",
        "Fee Structure",
        "Admission Process",
        "Contact Us",
    ],
    "fees_structure": [
        "Scholarship Info",
        "Apply Now",
        "Documents Needed",
        "Contact Us",
    ],
    "last_date": [
        "Admission Process",
        "Documents Needed",
        "Contact Us",
        "Book Appointment",
    ],
    "documents_needed": [
        "Admission Process",
        "Fee Structure",
        "Contact Us",
        "Book Appointment",
    ],
    "hostel_info": [
        "Fee Structure",
        "Contact Us",
        "Admission Process",
        "Book Appointment",
    ],
    "contact_info": [
        "Admission Process",
        "Book Appointment",
        "Courses Offered",
        "Fee Structure",
    ],
    "scholarship": [
        "Eligibility Criteria",
        "Fee Structure",
        "Admission Process",
        "Contact Us",
    ],
    "exam_schedule": [
        "Admission Process",
        "Last Date to Apply",
        "Result Status",
        "Eligibility",
    ],
    "result_status": [
        "Counselling Slots",
        "Documents Needed",
        "Contact Us",
        "Fee Structure",
    ],
    "slot_booking": [
        "Fee Structure",
        "Courses Offered",
        "Contact Us",
    ],
    "goodbye": [],
    "fallback": [
        "Admission Process",
        "Courses Offered",
        "Fee Structure",
        "Contact Us",
    ],
    "angry_user": ["Contact Us", "Book Appointment"],
}

FALLBACK_RESPONSES = [
    (
        "I'm sorry, I didn't quite understand that. Could you please rephrase? "
        "I can help with admissions, courses, fees, eligibility, scholarships, and more! 😊"
    ),
    (
        "Hmm, I'm not sure I got that. Try asking about Admission Process, "
        "Fee Structure, Courses, or Scholarship! 🎓"
    ),
    (
        "I didn't catch that. Could you ask again? "
        "I'm here to help with everything about Sunrise Institute of Technology! 🌅"
    ),
]

ANGRY_RESPONSE = (
    "I sincerely apologize for the inconvenience! 🙏 "
    "I understand your frustration. Please feel free to call us directly at "
    "**+91-755-2345678** or email **info@sunriseinstitute.ac.in** — "
    "our human counsellors will be happy to assist you."
)

OUT_OF_SCOPE_RESPONSE = (
    "That's an interesting question, but it's a bit outside my area of expertise! 😄 "
    "I'm specialized in helping you with information about Sunrise Institute of Technology. "
    "Try asking about admissions, courses, fees, or scholarships!"
)


class ResponseGenerator:
    """
    Generates chatbot responses by combining intent classification results,
    entity data, conversation context, and knowledge base information.
    """

    def __init__(self):
        """Load knowledge base and intents on initialization."""
        self.kb = self._load_kb()
        logger.info("ResponseGenerator initialized.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        intent: str,
        entities: dict,
        context: dict,
        slot_state: str = "IDLE",
        extra_data: dict | None = None,
    ) -> tuple[str, list[str]]:
        """
        Generate a response string and a list of quick-reply suggestions.

        Args:
            intent: Classified intent tag.
            entities: Extracted entities for this turn.
            context: Session context dict from ContextManager.
            slot_state: Current slot-filling state string.
            extra_data: Optional extra dict (e.g., slot_prompt from SlotManager).

        Returns:
            Tuple of (response text, list of quick-reply strings).
        """
        extra_data = extra_data or {}

        # Slot prompt takes priority during booking flow
        if slot_state not in ("IDLE", "COMPLETED", "CANCELLED") and extra_data.get("slot_prompt"):
            return extra_data["slot_prompt"], []

        handler = self._get_handler(intent)
        response = handler(entities, context, extra_data)
        quick_replies = QUICK_REPLIES.get(intent, QUICK_REPLIES["fallback"])

        return response, quick_replies

    # ------------------------------------------------------------------
    # Intent response handlers
    # ------------------------------------------------------------------

    def _get_handler(self, intent: str):
        """Return the appropriate response handler for the given intent."""
        handlers = {
            "greeting": self._handle_greeting,
            "goodbye": self._handle_goodbye,
            "admission_process": self._handle_admission_process,
            "courses_offered": self._handle_courses_offered,
            "eligibility": self._handle_eligibility,
            "fees_structure": self._handle_fees_structure,
            "last_date": self._handle_last_date,
            "documents_needed": self._handle_documents_needed,
            "hostel_info": self._handle_hostel_info,
            "contact_info": self._handle_contact_info,
            "scholarship": self._handle_scholarship,
            "exam_schedule": self._handle_exam_schedule,
            "result_status": self._handle_result_status,
            "slot_booking": self._handle_slot_booking,
            "angry_user": self._handle_angry,
            "fallback": self._handle_fallback,
        }
        return handlers.get(intent, self._handle_fallback)

    def _handle_greeting(self, entities, context, extra):
        greetings = [
            "Hello! 👋 Welcome to **Sunrise Institute of Technology, Bhopal**! "
            "How can I help you today? Ask me about admissions, courses, fees, or scholarships!",
            "Hi there! 🎓 I'm the Sunrise Institute chatbot. "
            "What would you like to know? Admissions, Fees, Courses, or something else?",
            "Namaste! 🙏 Welcome to Sunrise Institute of Technology. "
            "I'm here to guide you through your admission journey. What's on your mind?",
        ]
        return random.choice(greetings)

    def _handle_goodbye(self, entities, context, extra):
        goodbyes = [
            "Thank you for chatting with us! 🌅 Best of luck with your admission. "
            "Feel free to reach out anytime. Visit us at **Sunrise Institute of Technology, Bhopal**!",
            "Goodbye! 👋 Hope to see you on campus soon. "
            "For more info, visit **www.sunriseinstitute.ac.in**.",
            "Take care! 😊 If you have more questions, we're always here. "
            "All the best for your future!",
        ]
        return random.choice(goodbyes)

    def _handle_admission_process(self, entities, context, extra):
        steps = self.kb.get("admission_process", [])
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        return (
            f"📋 **Admission Process at Sunrise Institute of Technology:**\n\n"
            f"{steps_text}\n\n"
            f"For details, visit **www.sunriseinstitute.ac.in** or call **+91-755-2345678**."
        )

    def _handle_courses_offered(self, entities, context, extra):
        ug = self.kb["courses"]["ug"]
        pg = self.kb["courses"]["pg"]
        ug_list = "\n".join(
            f"  • {c['short']} ({c['duration']}) — ₹{c['fees_per_year']:,}/year" for c in ug
        )
        pg_list = "\n".join(
            f"  • {c['short']} ({c['duration']}) — ₹{c['fees_per_year']:,}/year" for c in pg
        )
        return (
            f"🎓 **Courses Offered at Sunrise Institute of Technology:**\n\n"
            f"**Undergraduate (UG):**\n{ug_list}\n\n"
            f"**Postgraduate (PG):**\n{pg_list}\n\n"
            f"Ask me for details about any specific course!"
        )

    def _handle_eligibility(self, entities, context, extra):
        course_name = entities.get("course") or context.get("entities_so_far", {}).get("course")
        if course_name:
            course_info = self._find_course(course_name)
            if course_info:
                return (
                    f"📚 **Eligibility for {course_info['short']}:**\n\n"
                    f"✅ {course_info['eligibility']}\n"
                    f"📝 Entrance Exam: {course_info['entrance_exam']}\n"
                    f"⏱ Duration: {course_info['duration']}\n"
                    f"💺 Available Seats: {course_info['seats']}"
                )

        # Generic eligibility summary
        return (
            "📋 **Eligibility Criteria at Sunrise Institute of Technology:**\n\n"
            "• **B.Tech (CS/EC/ME/CE):** 60% in PCM in 12th + JEE Mains / SITEE\n"
            "• **BCA:** 55% in 12th with Mathematics\n"
            "• **BBA / B.Com:** 50% in 12th (any stream)\n"
            "• **B.Sc:** 55% in 12th (Science stream)\n"
            "• **M.Tech:** B.Tech/B.E. with 60% + GATE (preferred)\n"
            "• **MBA:** Graduation with 50% + CAT/MAT/XAT\n"
            "• **MCA:** BCA/B.Sc with 55% + Maths\n"
            "• **M.Sc:** Relevant B.Sc with 55%\n\n"
            "Which course eligibility would you like to know more about?"
        )

    def _handle_fees_structure(self, entities, context, extra):
        course_name = entities.get("course") or context.get("entities_so_far", {}).get("course")
        if course_name:
            course_info = self._find_course(course_name)
            if course_info:
                return (
                    f"💰 **Fee Structure for {course_info['short']}:**\n\n"
                    f"• Annual Fee: **₹{course_info['fees_per_year']:,}** per year\n"
                    f"• Duration: {course_info['duration']}\n"
                    f"• Total Course Fee: "
                    f"₹{course_info['fees_per_year'] * int(course_info['duration'].split()[0]):,} "
                    f"(approx.)\n\n"
                    f"💡 Scholarships available! Ask me about scholarship opportunities."
                )

        # Full fee table
        ug = self.kb["courses"]["ug"]
        pg = self.kb["courses"]["pg"]
        ug_fees = "\n".join(
            f"  • {c['short']}: **₹{c['fees_per_year']:,}**/year" for c in ug
        )
        pg_fees = "\n".join(
            f"  • {c['short']}: **₹{c['fees_per_year']:,}**/year" for c in pg
        )
        return (
            f"💰 **Fee Structure at Sunrise Institute of Technology:**\n\n"
            f"**UG Courses:**\n{ug_fees}\n\n"
            f"**PG Courses:**\n{pg_fees}\n\n"
            f"💡 Scholarships available for merit (>90%), EWS, SC/ST, and Sports categories!"
        )

    def _handle_last_date(self, entities, context, extra):
        dates = self.kb["important_dates"]
        return (
            f"📅 **Important Dates for {dates['academic_year']} Admissions:**\n\n"
            f"• 📝 Form Start Date: **{dates['form_start_date']}**\n"
            f"• ⏰ Form Last Date: **{dates['form_last_date']}**\n"
            f"• 📋 Application Fee: ₹{dates['application_fee']}\n"
            f"• 🎓 Entrance Exam: **{dates['entrance_exam_date']}**\n"
            f"• 🪪 Admit Card Release: {dates['admit_card_release']}\n"
            f"• 📊 Result Declaration: **{dates['result_declaration']}**\n"
            f"• 📋 First Merit List: {dates['first_merit_list']}\n"
            f"• 🗣 Counselling: {dates['counselling_start']} – {dates['counselling_end']}\n"
            f"• 🏫 Classes Begin: **{dates['classes_begin']}**\n\n"
            f"Don't miss the deadline! Apply at **www.sunriseinstitute.ac.in**"
        )

    def _handle_documents_needed(self, entities, context, extra):
        docs = self.kb["documents_required"]
        docs_text = "\n".join(f"{i+1}. {doc}" for i, doc in enumerate(docs))
        return (
            f"📄 **Documents Required for Admission:**\n\n"
            f"{docs_text}\n\n"
            f"⚠️ All documents must be **self-attested**. "
            f"Bring originals + 2 sets of photocopies."
        )

    def _handle_hostel_info(self, entities, context, extra):
        h = self.kb["hostel"]
        bh = h["boys_hostel"]
        gh = h["girls_hostel"]
        bh_facilities = ", ".join(bh["facilities"])
        gh_facilities = ", ".join(gh["facilities"])
        return (
            f"🏠 **Hostel Information:**\n\n"
            f"**Boys Hostel ({bh['name']}):**\n"
            f"• Capacity: {bh['capacity']} students\n"
            f"• Annual Fee: **₹{bh['fees_per_year']:,}** ({bh['includes']})\n"
            f"• Room Types: {', '.join(bh['room_types'])}\n"
            f"• Facilities: {bh_facilities}\n\n"
            f"**Girls Hostel ({gh['name']}):**\n"
            f"• Capacity: {gh['capacity']} students\n"
            f"• Annual Fee: **₹{gh['fees_per_year']:,}** ({gh['includes']})\n"
            f"• Room Types: {', '.join(gh['room_types'])}\n"
            f"• Facilities: {gh_facilities}\n\n"
            f"📧 Contact: **{h['contact']}** | 📞 Warden: {h['warden_phone']}"
        )

    def _handle_contact_info(self, entities, context, extra):
        c = self.kb["college"]
        phones = " / ".join(c["phone"])
        return (
            f"📞 **Contact Sunrise Institute of Technology:**\n\n"
            f"• 📱 Phone: **{phones}**\n"
            f"• 📧 Email: **{c['email']}**\n"
            f"• 📧 Admissions: {c['admissions_email']}\n"
            f"• 🌐 Website: **{c['website']}**\n"
            f"• 📍 Address: {c['address']}\n"
            f"• ⏰ Office Hours: **{c['office_hours']}**\n"
            f"• 🗺 Google Maps: {c['google_maps']}"
        )

    def _handle_scholarship(self, entities, context, extra):
        scholarships = self.kb["scholarships"]
        lines = []
        for i, s in enumerate(scholarships, 1):
            lines.append(
                f"{i}. **{s['name']}:** {s['description']}\n"
                f"   💰 Benefit: {s['benefit']}\n"
                f"   ✅ Eligibility: {s['eligibility']}"
            )
        return (
            f"🏆 **Scholarship Opportunities at Sunrise Institute:**\n\n"
            + "\n\n".join(lines)
            + "\n\n📝 Apply for scholarship within **30 days of admission**. "
            "Contact **admissions@sunriseinstitute.ac.in** for details."
        )

    def _handle_exam_schedule(self, entities, context, extra):
        dates = self.kb["important_dates"]
        centers = ", ".join(dates["exam_centers"])
        return (
            f"📝 **Entrance Exam Details (SITEE {dates['academic_year']}):**\n\n"
            f"• 📅 Date: **{dates['entrance_exam_date']}**\n"
            f"• ⏰ Duration: **3 hours**\n"
            f"• 📋 Pattern: **MCQ (100 questions)**\n"
            f"• 📚 Subjects: Physics, Chemistry, Maths/Biology (as applicable)\n"
            f"• 💯 Total Marks: **400** (4 marks per correct answer, -1 for wrong)\n"
            f"• 🪪 Admit Card: Available from **{dates['admit_card_release']}**\n"
            f"• 🏙 Exam Centers: {centers}\n"
            f"• 📖 Syllabus: NCERT 11th & 12th standard\n\n"
            f"Download admit card from **www.sunriseinstitute.ac.in/admit-card**"
        )

    def _handle_result_status(self, entities, context, extra):
        dates = self.kb["important_dates"]
        return (
            f"📊 **Result & Merit List Information:**\n\n"
            f"• 📅 Entrance Result: **{dates['result_declaration']}**\n"
            f"• 📋 First Merit List: **{dates['first_merit_list']}**\n"
            f"• 📋 Second Merit List: **{dates['second_merit_list']}** (if required)\n"
            f"• 🗣 Counselling: {dates['counselling_start']} – {dates['counselling_end']}\n\n"
            f"🔍 Check results at: **www.sunriseinstitute.ac.in/results**\n"
            f"You'll need your **roll number** and **date of birth**.\n"
            f"📱 SMS & email notifications will be sent to registered candidates."
        )

    def _handle_slot_booking(self, entities, context, extra):
        return (
            "📅 I'll help you book a counselling appointment at Sunrise Institute! "
            "Let me collect a few details from you."
        )

    def _handle_angry(self, entities, context, extra):
        return ANGRY_RESPONSE

    def _handle_fallback(self, entities, context, extra):
        return random.choice(FALLBACK_RESPONSES)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_kb(self) -> dict:
        """Load knowledge_base.json from disk."""
        try:
            with open(KB_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.error("Failed to load knowledge base: %s", exc)
            return {}

    def _find_course(self, course_name: str) -> dict | None:
        """
        Find a course dict in the knowledge base by partial name match.

        Args:
            course_name: Course name or short form to search for.

        Returns:
            Course dict if found, else None.
        """
        course_lower = course_name.lower()
        for level in ("ug", "pg"):
            for course in self.kb["courses"][level]:
                if (
                    course_lower in course["short"].lower()
                    or course_lower in course["name"].lower()
                    or course["short"].lower() in course_lower
                ):
                    return course
        return None
