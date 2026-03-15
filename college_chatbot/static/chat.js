/**
 * chat.js — Sunrise Institute College Enquiry Chatbot
 * Handles all client-side chat interaction with zero dependencies.
 */

// ── Session ID ───────────────────────────────────────────────────────────────

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === 'x' ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
}

const SESSION_KEY = 'sit_session_id';
let sessionId = localStorage.getItem(SESSION_KEY);
if (!sessionId) {
  sessionId = generateUUID();
  localStorage.setItem(SESSION_KEY, sessionId);
}

// ── DOM References ────────────────────────────────────────────────────────────

const chatWindow     = document.getElementById('chatWindow');
const userInput      = document.getElementById('userInput');
const sendBtn        = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');

// ── Helpers ───────────────────────────────────────────────────────────────────

function getTime() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/** Render basic markdown: **bold**, \n line breaks */
function renderMarkdown(text) {
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\n/g, '<br/>');
}

/** Scroll chat window to the bottom */
function scrollToBottom() {
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

// ── Message Rendering ─────────────────────────────────────────────────────────

/**
 * Append a message bubble to the chat window.
 * @param {string} text       - Message content (may contain **bold** and \n)
 * @param {'bot'|'user'} role - Sender role
 * @param {string[]} quickReplies - Optional quick reply button labels
 */
function appendMessage(text, role, quickReplies = []) {
  const row = document.createElement('div');
  row.className = `message-row ${role}`;

  // Bot avatar (only for bot)
  if (role === 'bot') {
    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.textContent = 'SI';
    row.appendChild(avatar);
  }

  // Bubble wrapper
  const wrap = document.createElement('div');
  wrap.className = 'bubble-wrap';

  // Bubble
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = renderMarkdown(text);
  wrap.appendChild(bubble);

  // Timestamp
  const ts = document.createElement('div');
  ts.className = 'timestamp';
  ts.textContent = getTime();
  wrap.appendChild(ts);

  // Quick reply buttons
  if (quickReplies && quickReplies.length > 0) {
    const qrContainer = document.createElement('div');
    qrContainer.className = 'quick-replies';
    quickReplies.forEach((label) => {
      const btn = document.createElement('button');
      btn.className = 'quick-btn';
      btn.textContent = label;
      btn.addEventListener('click', () => {
        sendUserMessage(label);
      });
      qrContainer.appendChild(btn);
    });
    wrap.appendChild(qrContainer);
  }

  row.appendChild(wrap);
  chatWindow.appendChild(row);
  scrollToBottom();
  return row;
}

// ── Typing Indicator ──────────────────────────────────────────────────────────

function showTyping() {
  typingIndicator.style.display = 'flex';
  scrollToBottom();
}

function hideTyping() {
  typingIndicator.style.display = 'none';
}

// ── Input State ───────────────────────────────────────────────────────────────

function setInputEnabled(enabled) {
  userInput.disabled = !enabled;
  sendBtn.disabled   = !enabled;
  if (enabled) userInput.focus();
}

// ── API Communication ─────────────────────────────────────────────────────────

/**
 * Send a message to the /chat endpoint and handle the response.
 * @param {string} text - Message to send.
 */
async function sendUserMessage(text) {
  const trimmed = text.trim();
  if (!trimmed) return;

  // Show user bubble
  appendMessage(trimmed, 'user');
  userInput.value = '';
  setInputEnabled(false);

  // Simulate typing delay for realism
  await new Promise((r) => setTimeout(r, 350));
  showTyping();

  try {
    const response = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: trimmed, session_id: sessionId }),
    });

    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const data = await response.json();

    hideTyping();

    // Main reply
    const reply = data.reply || "I'm sorry, something went wrong. Please try again.";
    const qr    = data.quick_replies || [];
    appendMessage(reply, 'bot', qr);

    // If booking completed, show booking ID highlight
    if (data.booking_id) {
      const bookingMsg = `✅ Your **Booking ID** is: **${data.booking_id}**\nPlease save this for reference.`;
      appendMessage(bookingMsg, 'bot', ['Track Appointment', 'Contact Us']);
    }

  } catch (err) {
    hideTyping();
    console.error('Chat API error:', err);
    appendMessage(
      '⚠️ Network error. Please check your connection and try again.',
      'bot',
      ['Retry', 'Contact Us']
    );
  } finally {
    setInputEnabled(true);
  }
}

// ── Send Handler (button + Enter key) ────────────────────────────────────────

function sendMessage() {
  const text = userInput.value.trim();
  if (text) sendUserMessage(text);
}

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

// Disable send button when input is empty
userInput.addEventListener('input', () => {
  sendBtn.disabled = userInput.value.trim().length === 0;
});

// ── Welcome Message ───────────────────────────────────────────────────────────

(function showWelcome() {
  const welcomeText =
    '👋 Welcome to **Sunrise Institute of Technology, Bhopal**!\n\n' +
    'I\'m your virtual admission counsellor. I can help you with:\n' +
    '📚 Courses & Eligibility\n' +
    '💰 Fee Structure & Scholarships\n' +
    '📝 Admission Process & Dates\n' +
    '📅 Book a Campus Visit\n\n' +
    'What would you like to know? 😊';

  appendMessage(welcomeText, 'bot', [
    'Admission Process',
    'Courses Offered',
    'Fee Structure',
    'Book Appointment',
  ]);

  // Initial send button state
  sendBtn.disabled = true;
  userInput.focus();
})();
