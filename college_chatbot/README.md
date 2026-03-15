# Sunrise Institute of Technology — College Enquiry Chatbot

A production-ready College Enquiry Chatbot built with **Python**, **NLTK**, **Flask**, and **SQLite**.

---

## 📁 Project Structure

```
college_chatbot/
├── app.py                     # Flask application entry point
├── chatbot/
│   ├── __init__.py
│   ├── nlp_engine.py          # TF-IDF vectorizer, cosine similarity, keyword boost
│   ├── intent_classifier.py   # Hybrid intent classification (exact + TF-IDF + keywords)
│   ├── entity_extractor.py    # Named entity extraction (person, phone, email, course, date)
│   ├── slot_manager.py        # State machine for appointment booking
│   ├── context_manager.py     # Multi-turn conversation context & pronoun resolution
│   └── response_generator.py  # Template-based response builder with quick replies
├── data/
│   ├── intents.json           # 15 intent categories with 12+ patterns each
│   ├── knowledge_base.json    # College info, courses, fees, dates, hostel, scholarships
│   └── faqs.json              # Common FAQs
├── models/
│   └── vectorizer.pkl         # Persisted TF-IDF model (auto-trained at startup)
├── database/
│   ├── college.db             # SQLite database (auto-created)
│   └── db_manager.py          # DB CRUD functions
├── templates/
│   ├── index.html             # Chat UI (navy + gold, responsive)
│   └── admin.html             # Admin panel
├── static/
│   ├── style.css              # CSS with variables, animations
│   └── chat.js                # Vanilla JS chat client
├── logs/                      # App and chat logs (auto-created)
├── requirements.txt
├── setup.py                   # One-shot setup script
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone / navigate to the project

```bash
cd college_chatbot
```

### 2. Install dependencies (VS Code terminal)

Run this command in the VS Code terminal (or any terminal) from inside the
`college_chatbot/` folder to install all required packages:

```bash
pip install -r requirements.txt
```

> **Tip:** If you have multiple Python versions installed, use `pip3` instead:
> ```bash
> pip3 install -r requirements.txt
> ```
> Or target a specific Python interpreter:
> ```bash
> python -m pip install -r requirements.txt
> ```

### 3. Run full setup (downloads NLTK data, inits DB, trains model)

```bash
python setup.py
```

> `setup.py` already runs `pip install -r requirements.txt` internally, so
> you can skip Step 2 and jump straight here if you prefer a single command.

### 4. Start the server

```bash
python app.py
```

### 5. Open your browser

```
http://localhost:5000
```

---

## 🤖 Features

| Feature | Details |
|---|---|
| **15 Intents** | Greeting, Admission, Courses, Fees, Eligibility, Documents, Hostel, Contact, Scholarship, Exam, Result, Slot Booking, Fallback, and more |
| **Hybrid NLP** | Exact match → TF-IDF cosine similarity → Keyword fallback |
| **Entity Extraction** | Person name, phone, email, course, date, time, city |
| **Slot Filling** | 8-state machine for booking counselling appointments |
| **Context Manager** | 10-turn history, pronoun resolution (uska, iska, that course) |
| **Multi-language** | Handles Hinglish: "fees kitni hai", "admission kaise hoga" |
| **Edge Cases** | Typos, short queries, angry users, out-of-scope, slot interruption |
| **Admin Panel** | `/admin` (password-protected) with stats, appointment management, CSV export |
| **SQLite DB** | Appointments, enquiry logs, and leads tables |

---

## 🔑 Admin Panel

Visit `http://localhost:5000/admin`

Default credentials:
- **Username:** `admin`
- **Password:** `sunrise2025`

Override via environment variables:
```bash
export ADMIN_USERNAME=myuser
export ADMIN_PASSWORD=mysecretpassword
```

---

## 🌐 API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI |
| `POST` | `/chat` | Main chatbot endpoint |
| `POST` | `/slot/cancel` | Cancel active booking |
| `GET` | `/health` | Health check |
| `GET` | `/admin` | Admin panel (Basic Auth) |
| `POST` | `/admin/appointment/<id>/confirm` | Confirm appointment |
| `POST` | `/admin/appointment/<id>/cancel` | Cancel appointment |
| `GET` | `/admin/export/appointments` | Download CSV |

### POST /chat

**Request:**
```json
{ "message": "B.Tech ki fees kitni hai?", "session_id": "uuid-here" }
```

**Response:**
```json
{
  "reply": "💰 Fee Structure for B.Tech CS: ...",
  "quick_replies": ["Scholarship Info", "Apply Now"],
  "intent": "fees_structure",
  "confidence": 0.87,
  "slot_state": "IDLE",
  "booking_id": null
}
```

---

## ⚙️ Configuration

| Environment Variable | Default | Description |
|---|---|---|
| `FLASK_SECRET_KEY` | Random hex | Flask session secret |
| `ADMIN_USERNAME` | `admin` | Admin panel username |
| `ADMIN_PASSWORD` | `sunrise2025` | Admin panel password |

---

## 📦 Dependencies

```
flask==3.0.0
flask-cors==4.0.0
nltk==3.9.3
scikit-learn==1.4.0
numpy==1.26.0
joblib==1.3.2
```

---

## 🏫 About Sunrise Institute of Technology

- **Location:** Bhopal, Madhya Pradesh
- **Established:** 2005
- **NAAC Grade:** A+
- **Affiliation:** RGPV, Bhopal
- **Courses:** B.Tech (CS, EC, ME, CE), BCA, BBA, B.Sc, B.Com, M.Tech, MCA, MBA, M.Sc
- **Placement Rate:** 94% | Avg Package: ₹6.8 LPA | Highest: ₹24 LPA

---

*Built with ❤️ using Python, NLTK, Flask, and SQLite.*
