
import os
import requests
import streamlit as st

from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from google.genai import types


# =====================================
# 1. CONFIGURATION
# =====================================

load_dotenv()

st.set_page_config(
    page_title="AI Exam Preparation Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

def get_secret(name):
    try:
        value = st.secrets.get(name)
        if value:
            return value
    except Exception:
        pass

    return os.getenv(name)


GOOGLE_API_KEY = get_secret("GOOGLE_API_KEY")
SERPER_API_KEY = get_secret("SERPER_API_KEY")

if not GOOGLE_API_KEY:
    st.error("Google API key missing in Streamlit Secrets.")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

MODEL = "gemini-3.5-flash"


# =====================================
# 2. CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.main-title {
    font-size: 34px;
    font-weight: 800;
    text-align: center;
    color: #8B9CFF;
}

.subtitle {
    text-align: center;
    color: #AAAAAA;
    margin-bottom: 30px;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    min-height: 42px;
}

.stChatInput {
    border-radius: 12px;
}

@media(max-width:600px) {
    .main-title {
        font-size: 24px;
    }

    .block-container {
        padding-left: 12px;
        padding-right: 12px;
    }
}

</style>
""", unsafe_allow_html=True)


# =====================================
# 3. SESSION STATE
# =====================================

defaults = {
    "messages": [],
    "paper_text": "",
    "paper_results": [],
    "pending_prompt": ""
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =====================================
# 4. AI SYSTEM PROMPT
# =====================================

SYSTEM_PROMPT = """
You are an Advanced AI Exam Preparation Agent.

Your main purpose is to help students prepare for exams.

YOUR RESPONSIBILITIES:

1. Find relevant past papers.
2. Analyze uploaded past paper PDFs.
3. Identify repeated questions.
4. Generate important short questions.
5. Generate important long questions.
6. Generate MCQs with answers.
7. Explain difficult concepts.
8. Analyze historical question frequency.
9. Suggest important topics for preparation.

PAST PAPER ANALYSIS:

- Analyze actual available paper content.
- Group similar questions by concept.
- Count how frequently questions appear.
- Mention the number of papers analyzed.
- Never invent historical statistics.
- Never claim a question is guaranteed to appear.
- Clearly separate historical frequency from predictions.
- If insufficient papers are available, explain the limitation.

RESPONSE FORMAT:

1. Past Paper Overview
2. Repeated Questions
3. Important Short Questions
4. Important Long Questions
5. Important MCQs
6. Historical Frequency
7. Estimated Importance
8. Preparation Strategy

Use simple language and clear headings.
Respond in the language used by the student.
"""


# =====================================
# 5. GOOGLE SEARCH
# =====================================

def google_search(query):

    if not SERPER_API_KEY:
        return []

    try:

        response = requests.post(
            "https://google.serper.dev/search",

            headers={
                "X-API-KEY": SERPER_API_KEY,
                "Content-Type": "application/json"
            },

            json={
                "q": query,
                "num": 10
            },

            timeout=30
        )

        response.raise_for_status()

        results = response.json().get("organic", [])

        return [
            {
                "title": item.get("title", "Unknown"),
                "link": item.get("link", ""),
                "snippet": item.get("snippet", "")
            }
            for item in results
        ]

    except requests.RequestException as e:

        st.error(f"Google Search Error: {e}")
        return []


# =====================================
# 6. PDF READER
# =====================================

def extract_pdf_text(files):

    combined_text = ""

    for file in files:

        try:

            reader = PdfReader(file)

            for number, page in enumerate(reader.pages):

                text = page.extract_text() or ""

                combined_text += f"""
                File: {file.name}
                Page: {number + 1}

                {text}
                """

        except Exception as e:

            st.warning(f"PDF Error: {e}")

    return combined_text[:50000]


# =====================================
# 7. GEMINI AI FUNCTION
# =====================================

def ask_gemini(prompt, history, academic_context):

    conversation = ""

    for message in history[-10:]:

        role = (
            "Student"
            if message["role"] == "user"
            else "Assistant"
        )

        conversation += f"""
        {role}: {message['content']}
        """

    full_prompt = f"""
    {SYSTEM_PROMPT}

    STUDENT ACADEMIC INFORMATION:

    {academic_context}

    PREVIOUS CHAT:

    {conversation}

    CURRENT QUESTION:

    {prompt}

    Give an accurate and helpful response.
    """

    response = client.models.generate_content(

        model=MODEL,

        contents=full_prompt,

        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=4000
        )
    )

    return response.text or "No response generated."


# =====================================
# 8. HEADER
# =====================================

st.markdown(
    '<div class="main-title">🎓 AI Exam Preparation Agent</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Your Smart AI Study Partner | Past Papers | Important Questions'
    '</div>',
    unsafe_allow_html=True
)


# =====================================
# 9. SIDEBAR
# =====================================

with st.sidebar:

    st.title("📚 Academic Dashboard")

    st.divider()

    board = st.selectbox(
        "Select Board / University",
        [
            "BISE Lahore",
            "BISE Gujranwala",
            "BISE Faisalabad",
            "BISE Multan",
            "BISE Rawalpindi",
            "Federal Board",
            "Punjab University",
            "Other",
            "NAT Entry Test"
        ]
    )

    if board == "Other":
        board = st.text_input("Enter Board Name")

    class_name = st.selectbox(
        "Select Class",
        [
            "9th Class",
            "10th Class",
            "1st Year",
            "2nd Year",
            "BS / University",
            "Entry Test"
        ]
    )

    subject = st.text_input(
        "Subject",
        placeholder="e.g. Physics, Computer Science"
    )

    chapter = st.text_input(
        "Chapter",
        placeholder="Optional"
    )

    st.divider()

    st.subheader("📂 Upload Past Papers")

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Analyze PDF Papers"):

        if uploaded_files:

            with st.spinner("Reading PDF Papers..."):

                st.session_state.paper_text = (
                    extract_pdf_text(uploaded_files)
                )

            if st.session_state.paper_text:
                st.success("PDF Analysis Data Ready!")
            else:
                st.warning("No readable text found in PDFs.")

        else:

            st.warning("Please upload PDF files.")

    if st.session_state.paper_text:

        st.success("Past Papers Loaded")

    st.divider()

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.rerun()

    if st.button("Reset PDF Data"):

        st.session_state.paper_text = ""
        st.session_state.paper_results = []
        st.rerun()


# =====================================
# 10. ACADEMIC DASHBOARD
# =====================================

st.subheader("📊 Academic Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Board", board)

c2.metric("Class", class_name)

c3.metric(
    "Subject",
    subject if subject else "Not Selected"
)

st.divider()


# =====================================
# 11. QUICK ACTIONS
# =====================================

st.subheader("⚡ Quick Actions")

col1, col2 = st.columns(2)

with col1:

    if st.button("📄 Find Past Papers"):

        st.session_state.pending_prompt = (
            "Find previous past papers and provide actual links."
        )

with col2:

    if st.button("⭐ Important Questions"):

        st.session_state.pending_prompt = (
            "Analyze uploaded papers and generate important "
            "short and long questions."
        )

col3, col4 = st.columns(2)

with col3:

    if st.button("📈 Frequency Analysis"):

        st.session_state.pending_prompt = (
            "Analyze historical frequency of questions "
            "in uploaded past papers."
        )

with col4:

    if st.button("📝 Generate MCQs"):

        st.session_state.pending_prompt = (
            "Generate important MCQs with correct answers "
            "from the selected subject and chapter."
        )


# =====================================
# 12. CHAT INTERFACE
# =====================================

st.divider()

st.subheader("💬 Chat With AI Assistant")

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# =====================================
# 13. CHAT INPUT
# =====================================

prompt = st.chat_input(
    "Ask anything about your subject..."
)

if st.session_state.pending_prompt:

    prompt = st.session_state.pending_prompt

    st.session_state.pending_prompt = ""


# =====================================
# 14. AI EXECUTION
# =====================================

if prompt:

    if not subject:

        st.warning("Please select your subject first.")

    else:

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        academic_context = f"""
        Board: {board}
        Class: {class_name}
        Subject: {subject}
        Chapter: {chapter}

        Uploaded Past Paper Text:

        {st.session_state.paper_text[:50000]}
        """

        search_keywords = [
            "past paper",
            "past papers",
            "previous paper",
            "previous papers",
            "find paper",
            "paper links"
        ]

        should_search = any(
            word in prompt.lower()
            for word in search_keywords
        )

        search_context = ""

        if should_search:

            query = (
                f"{board} {class_name} {subject} "
                f"{chapter} past papers PDF"
            )

            with st.spinner("Searching Google..."):

                results = google_search(query)

                st.session_state.paper_results = results

                for item in results:

                    search_context += f"""
                    Title: {item['title']}
                    Link: {item['link']}
                    Description: {item['snippet']}
                    """

        academic_context += f"""

        GOOGLE SEARCH RESULTS:

        {search_context}

        Search snippets are not full PDF papers.
        Do not claim to have analyzed a complete paper
        unless its actual text is available.
        """

        with st.chat_message("assistant"):

            with st.spinner("AI is analyzing..."):

                try:

                    answer = ask_gemini(
                        prompt,
                        st.session_state.messages[:-1],
                        academic_context
                    )

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:

                    st.error(f"Gemini Error: {e}")


# =====================================
# 15. SEARCH RESULTS
# =====================================

if st.session_state.paper_results:

    with st.expander("🔎 Google Search Results"):

        for item in st.session_state.paper_results:

            st.markdown(f"**{item['title']}**")

            st.write(item["snippet"])

            if item["link"]:

                st.markdown(
                    f"[Open Past Paper]({item['link']})"
                )

            st.divider()


# =====================================
# 16. FOOTER
# =====================================

st.divider()

st.caption(
    "AI Exam Preparation Agent | Powered by Gemini AI"
)

st.caption(
    "Historical question patterns are estimates, "
    "not guaranteed future exam questions."
)
