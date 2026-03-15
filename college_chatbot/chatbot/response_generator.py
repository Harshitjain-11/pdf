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

# Quick reply suggestions per intent (PART G — Dynamic System)
QUICK_REPLIES: dict[str, list[str]] = {
    "fees_structure": [
        "Scholarship Info", "Eligibility Criteria",
        "Documents Needed", "Apply Now",
    ],
    "courses_offered": [
        "B.Tech Details", "Fee Structure",
        "Placement Info", "Eligibility Criteria",
    ],
    "admission_process": [
        "Documents Needed", "Last Date to Apply",
        "Fee Structure", "Book Campus Visit",
    ],
    "eligibility": [
        "Fee Structure", "Admission Process",
        "Documents Needed", "Book Campus Visit",
    ],
    "placement_info": [
        "Courses Offered", "Fee Structure",
        "Contact Us", "Book Campus Visit",
    ],
    "documents_needed": [
        "Admission Process", "Fee Structure",
        "Last Date to Apply", "Book Campus Visit",
    ],
    "last_date": [
        "Admission Process", "Documents Needed",
        "Book Campus Visit", "Contact Us",
    ],
    "scholarship": [
        "Eligibility Criteria", "Fee Structure",
        "Admission Process", "Contact Us",
    ],
    "hostel_info": [
        "Fee Structure", "Contact Us",
        "Book Campus Visit", "Courses Offered",
    ],
    "slot_booking": ["Cancel Booking"],
    "greeting": [
        "Courses Offered", "Fee Structure",
        "Admission Process", "Book Appointment",
    ],
    "contact_info": [
        "Book Campus Visit", "Admission Process",
        "Courses Offered", "Fee Structure",
    ],
    "campus_life": [
        "Fee Structure", "Placement Info",
        "Contact Us", "Book Campus Visit",
    ],
    "lateral_entry": [
        "Eligibility Criteria", "Documents Needed",
        "Fee Structure", "Contact Us",
    ],
    "fallback": [
        "Courses Offered", "Fee Structure",
        "Admission Process", "Contact Us",
    ],
    "angry_user": ["Contact Us", "Book Campus Visit"],
    "out_of_scope": [
        "Courses Offered", "Fee Structure",
        "Admission Process", "Book Appointment",
    ],
    "exam_schedule": [
        "Admission Process", "Last Date to Apply",
        "Eligibility Criteria", "Contact Us",
    ],
    "result_status": [
        "Documents Needed", "Fee Structure",
        "Contact Us", "Book Campus Visit",
    ],
    "goodbye": [],
    "naac_ranking": [
        "Courses Offered", "Placement Info",
        "Fee Structure", "Contact Us",
    ],
    "anti_ragging": [
        "Contact Us", "Admission Process",
        "Courses Offered", "Book Campus Visit",
    ],
    "refund_policy": [
        "Fee Structure", "Admission Process",
        "Contact Us", "Documents Needed",
    ],
    "nri_quota": [
        "Fee Structure", "Eligibility Criteria",
        "Admission Process", "Contact Us",
    ],
    "events_fest": [
        "Courses Offered", "Placement Info",
        "Contact Us", "Book Campus Visit",
    ],
    "comparison": [
        "Courses Offered", "Placement Info",
        "Fee Structure", "Book Campus Visit",
    ],
    "faculty_info": [
        "Courses Offered", "Placement Info",
        "Fee Structure", "Contact Us",
    ],
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
        "I'm here to help with everything about Institute of Technology & Management, Gwalior! 🎓"
    ),
]

ANGRY_RESPONSE = (
    "Mujhe bahut afsos hai ki aapko accha experience nahi "
    "mila. 😔 Mein abhi bhi poori koshish karoonga aapki "
    "help karne ki. Ya phir directly humse baat karein:\n"
    "📞 +91-751-2970300\n📧 info@itmgwalior.ac.in"
)

OUT_OF_SCOPE_RESPONSE = (
    "Yeh meri expertise se thoda bahar hai! 😅 Mein sirf "
    "ITM Gwalior ke admissions, courses, placements aur "
    "campus life ke baare mein help kar sakta hoon.\n\n"
    "Kya mein inn topics mein se kisi mein help karoon?"
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
            "out_of_scope": self._handle_out_of_scope,
            "placement_info": self._handle_placement_info,
            "campus_life": self._handle_campus_life,
            "faculty_info": self._handle_faculty_info,
            "lateral_entry": self._handle_lateral_entry,
            "naac_ranking": self._handle_naac_ranking,
            "anti_ragging": self._handle_anti_ragging,
            "refund_policy": self._handle_refund_policy,
            "nri_quota": self._handle_nri_quota,
            "events_fest": self._handle_events_fest,
            "comparison": self._handle_comparison,
            "fallback": self._handle_fallback,
        }
        return handlers.get(intent, self._handle_fallback)

    def _handle_greeting(self, entities, context, extra):
        greetings = [
            "Hello! 👋 Welcome to **Institute of Technology & Management, Gwalior**! "
            "How can I help you today? Ask me about admissions, courses, fees, or scholarships!",
            "Hi there! 🎓 I'm the ITM Gwalior chatbot. "
            "What would you like to know? Admissions, Fees, Courses, or something else?",
            "Namaste! 🙏 Welcome to Institute of Technology & Management, Gwalior. "
            "I'm here to guide you through your admission journey. What's on your mind?",
        ]
        return random.choice(greetings)

    def _handle_goodbye(self, entities, context, extra):
        goodbyes = [
            "Thank you for chatting with us! 🎓 Best of luck with your admission. "
            "Feel free to reach out anytime. Visit us at **Institute of Technology & Management, Gwalior**!",
            "Goodbye! 👋 Hope to see you on campus soon. "
            "For more info, visit **www.itmgwalior.ac.in**.",
            "Take care! 😊 If you have more questions, we're always here. "
            "All the best for your future!",
        ]
        return random.choice(goodbyes)

    def _handle_admission_process(self, entities, context, extra):
        steps = self.kb.get("admission_process", [])
        steps_text = "\n".join(f"{i+1}. {step}" for i, step in enumerate(steps))
        return (
            f"📋 **Admission Process at Institute of Technology & Management, Gwalior:**\n\n"
            f"{steps_text}\n\n"
            f"For details, visit **www.itmgwalior.ac.in** or call **+91-751-2970300**."
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
            f"🎓 **Courses Offered at Institute of Technology & Management, Gwalior:**\n\n"
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
            "📋 **Eligibility Criteria at Institute of Technology & Management, Gwalior:**\n\n"
            "• **B.Tech (CS/EC/ME/CE):** 60% in PCM in 12th + JEE Mains (RGPV DTE MP counselling)\n"
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
            f"💰 **Fee Structure at Institute of Technology & Management, Gwalior:**\n\n"
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
            f"Don't miss the deadline! Apply at **www.itmgwalior.ac.in**"
        )

    def _handle_documents_needed(self, entities, context, extra):
        docs_detailed = self.kb.get("documents_detailed")
        if docs_detailed:
            mandatory = "\n".join(
                f"  {i+1}. {doc}" for i, doc in enumerate(docs_detailed["mandatory"])
            )
            conditional = "\n".join(
                f"  • {doc}" for doc in docs_detailed["conditional"]
            )
            notes = docs_detailed.get("notes", "")
            return (
                f"📁 ITM Gwalior Admission Documents:\n\n"
                f"✅ Mandatory Documents:\n{mandatory}\n\n"
                f"📎 Conditional (agar applicable ho):\n{conditional}\n\n"
                f"⚠️ {notes}"
            )
        # Fallback to simple list
        docs = self.kb.get("documents_required", [])
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
            f"📞 **Contact Institute of Technology & Management, Gwalior:**\n\n"
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
            f"🏆 **Scholarship Opportunities at ITM Gwalior:**\n\n"
            + "\n\n".join(lines)
            + "\n\n📝 Apply for scholarship within **30 days of admission**. "
            "Contact **admissions@itmgwalior.ac.in** for details."
        )

    def _handle_exam_schedule(self, entities, context, extra):
        dates = self.kb["important_dates"]
        centers = ", ".join(dates["exam_centers"])
        return (
            f"📝 **Entrance Exam Details — {dates['academic_year']} Admissions:**\n\n"
            f"• 📅 Date: **{dates['entrance_exam_date']}**\n"
            f"• ⏰ Duration: **3 hours**\n"
            f"• 📋 Pattern: **MCQ (100 questions)**\n"
            f"• 📚 Subjects: Physics, Chemistry, Maths/Biology (as applicable)\n"
            f"• 💯 Total Marks: **400** (4 marks per correct answer, -1 for wrong)\n"
            f"• 🪪 Admit Card: Available from **{dates['admit_card_release']}**\n"
            f"• 🏙 Exam Centers: {centers}\n"
            f"• 📖 Syllabus: NCERT 11th & 12th standard\n\n"
            f"B.Tech admissions follow JEE Mains / RGPV DTE MP counselling. "
            f"Download admit card from **www.itmgwalior.ac.in/admit-card**"
        )

    def _handle_result_status(self, entities, context, extra):
        dates = self.kb["important_dates"]
        return (
            f"📊 **Result & Merit List Information:**\n\n"
            f"• 📅 Entrance Result: **{dates['result_declaration']}**\n"
            f"• 📋 First Merit List: **{dates['first_merit_list']}**\n"
            f"• 📋 Second Merit List: **{dates['second_merit_list']}** (if required)\n"
            f"• 🗣 Counselling: {dates['counselling_start']} – {dates['counselling_end']}\n\n"
            f"🔍 Check results at: **www.itmgwalior.ac.in/results**\n"
            f"You'll need your **roll number** and **date of birth**.\n"
            f"📱 SMS & email notifications will be sent to registered candidates."
        )

    def _handle_slot_booking(self, entities, context, extra):
        return (
            "📅 I'll help you book a counselling appointment at ITM Gwalior! "
            "Let me collect a few details from you."
        )

    def _handle_angry(self, entities, context, extra):
        return ANGRY_RESPONSE

    def _handle_out_of_scope(self, entities, context, extra):
        return OUT_OF_SCOPE_RESPONSE

    def _handle_fallback(self, entities, context, extra):
        return random.choice(FALLBACK_RESPONSES)

    def _handle_placement_info(self, entities, context, extra):
        placements = self.kb.get("placements", {})
        if not placements:
            return "ITM Gwalior has excellent placement records. Contact placement cell for details."
        recruiters_list = "\n".join(
            f"  • {r['name']} — ₹{r['package_lpa']} LPA ({', '.join(r['roles'])})"
            for r in placements.get("top_recruiters", [])
        )
        branch_wise = "\n".join(
            f"  • {branch}: {pct}"
            for branch, pct in placements.get("branch_wise", {}).items()
        )
        return (
            f"🏢 ITM Gwalior Placement Record {placements.get('year', '')}:\n\n"
            f"• Overall Placement: {placements.get('overall_percentage', 'N/A')}\n"
            f"• Average Package: ₹{placements.get('average_package_lpa', 'N/A')} LPA\n"
            f"• Highest Package: ₹{placements.get('highest_package_lpa', 'N/A')} LPA\n\n"
            f"🌟 Top Recruiters:\n{recruiters_list}\n\n"
            f"📊 Branch-wise:\n{branch_wise}\n\n"
            f"Branch-wise placement bhi batayein?"
        )

    def _handle_campus_life(self, entities, context, extra):
        campus = self.kb.get("campus", {})
        if not campus:
            return "ITM Gwalior has a vibrant campus with modern facilities!"
        labs = ", ".join(campus.get("labs", [])[:6])
        sports = ", ".join(campus.get("sports", [])[:6])
        return (
            f"🏫 ITM Gwalior Campus Life:\n\n"
            f"• Area: {campus.get('area_acres', 'N/A')} acres\n"
            f"• WiFi: {campus.get('wifi', 'Available')}\n"
            f"• Library: {campus.get('library_books', 'N/A')}+ books\n"
            f"• Labs: {labs}\n"
            f"• Sports: {sports}\n"
            f"• NAAC Grade: {campus.get('naac_grade', 'N/A')}\n\n"
            f"Aur kuch jaanna hai campus ke baare mein?"
        )

    def _handle_faculty_info(self, entities, context, extra):
        campus = self.kb.get("campus", {})
        labs = ", ".join(campus.get("labs", [])) if campus else "Multiple labs"
        return (
            "👨‍🏫 ITM Gwalior Faculty Information:\n\n"
            "• Experienced faculty with PhD and industry background\n"
            "• Regular workshops and seminars\n"
            "• Research-oriented teaching methodology\n"
            f"• Labs: {labs}\n"
            "• Industry collaboration for practical exposure\n\n"
            "Kisi specific department ke baare mein jaanna hai?"
        )

    def _handle_lateral_entry(self, entities, context, extra):
        le = self.kb.get("lateral_entry", {})
        if not le:
            return "Lateral entry admission available hai ITM Gwalior mein. Details ke liye contact karein."
        return (
            "🎓 Lateral Entry (Direct 2nd Year) — ITM Gwalior:\n\n"
            f"• Eligibility: {le.get('eligibility', 'N/A')}\n"
            f"• Direct admission: {le.get('direct_admission_to', 'N/A')}\n"
            f"• Available seats: {le.get('seats', 'N/A')}\n"
            f"• Last date: {le.get('last_date', 'N/A')}\n"
            f"• Process: {le.get('process', 'N/A')}\n\n"
            "Documents aur process ke baare mein aur jaanna chahte ho?"
        )

    def _handle_naac_ranking(self, entities, context, extra):
        college = self.kb.get("college", {})
        campus = self.kb.get("campus", {})
        return (
            "🏆 ITM Gwalior Rankings & Accreditation:\n\n"
            f"• NAAC Grade: {college.get('naac_grade', 'N/A')}\n"
            f"• Affiliated to: {college.get('affiliating_university', 'RGPV Bhopal')}\n"
            f"• Type: {college.get('type', 'N/A')}\n"
            f"• Established: {college.get('established', 'N/A')}\n"
            f"• UGC Approved: {'Yes' if campus.get('ugc_approved') else 'Yes'}\n"
            f"• Accreditations: {', '.join(college.get('accreditations', []))}\n\n"
            "Aur kuch jaanna hai?"
        )

    def _handle_anti_ragging(self, entities, context, extra):
        ar = self.kb.get("anti_ragging", {})
        if not ar:
            return "ITM Gwalior has strict anti-ragging policy. Contact college for details."
        return (
            "🛡️ Anti-Ragging Policy — ITM Gwalior:\n\n"
            f"• Policy: {ar.get('policy', 'Zero tolerance')}\n"
            f"• Helpline: {ar.get('helpline', 'N/A')}\n"
            f"• College Committee: {ar.get('college_committee', 'N/A')}\n"
            f"• Email: {ar.get('email', 'N/A')}\n\n"
            "Campus bilkul safe hai! Koi bhi complaint ho toh turant action liya jaata hai."
        )

    def _handle_refund_policy(self, entities, context, extra):
        rp = self.kb.get("refund_policy", {})
        if not rp:
            return "Refund policy ke liye admission office se contact karein."
        return (
            "💰 Refund Policy — ITM Gwalior:\n\n"
            f"• Before July 15: {rp.get('before_july_15', 'N/A')}\n"
            f"• July 15 to Aug 1: {rp.get('july_15_to_aug_1', 'N/A')}\n"
            f"• After Aug 1: {rp.get('after_aug_1', 'N/A')}\n\n"
            "Aur kuch jaanna hai?"
        )

    def _handle_nri_quota(self, entities, context, extra):
        return (
            "🌏 NRI / Management Quota — ITM Gwalior:\n\n"
            "• NRI quota seats available hain selected courses mein\n"
            "• Management quota admission bhi available hai\n"
            "• Direct application through college admission office\n"
            "• Documents: Passport, visa, NRI certificate (if applicable)\n\n"
            "Details ke liye admissions office se contact karein:\n"
            "📞 +91-751-2970300\n"
            "📧 admissions@itmgwalior.ac.in"
        )

    def _handle_events_fest(self, entities, context, extra):
        return (
            "🎉 Events & Fests — ITM Gwalior:\n\n"
            "• 💻 Tech Fest: Annual technical festival with hackathons, coding contests\n"
            "• 🎭 Cultural Fest: Music, dance, drama, art competitions\n"
            "• 🏏 Sports Day: Annual sports meet with inter-college competitions\n"
            "• 🎓 Workshops & Seminars: Industry experts ke sessions\n"
            "• 🤝 Club Activities: Coding club, robotics club, literary club\n\n"
            "Campus life bahut vibrant hai! Aur kuch jaanna hai?"
        )

    def _handle_comparison(self, entities, context, extra):
        college = self.kb.get("college", {})
        placements = self.kb.get("placements", {})
        naac = college.get("naac_grade", "B++")
        pct = placements.get("overall_percentage", "85%")
        return (
            f"ITM Gwalior apne region ke top engineering colleges mein "
            f"se ek hai. NAAC {naac} grade, RGPV affiliated, "
            f"aur {pct} placement record ke saath "
            f"yeh ek solid choice hai. Kisi specific aspect ka "
            f"comparison chahiye?"
        )

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
