import os
import re
import asyncio
import tempfile
import xml.etree.ElementTree as ET

import edge_tts
import requests
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    try:
        OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]
    except Exception:
        OPENROUTER_API_KEY = None

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY is not configured.")
    st.stop()

st.set_page_config(
    page_title="MedAssist",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

SYSTEM_PROMPT = """
You are a medical information voice assistant.

Your job is to provide general educational medical information.

Important rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use information retrieved from the medical information tool.
- Since your response may be spoken aloud, keep the answer concise.
- Use simple language.
- Do not use Markdown.
- Avoid long lists.
- If the user describes an emergency or severe symptoms, advise them to seek immediate professional/emergency care.
"""

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
    .main {
        background: #f7f9fc;
    }

    .block-container {
        max-width: 1050px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .hero {
        padding: 1.8rem 2rem;
        border-radius: 22px;
        background: linear-gradient(135deg, #e8f4ff, #f5f9ff);
        border: 1px solid #d7e8f7;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.25rem;
        color: #16324f;
    }

    .hero p {
        margin: 0.5rem 0 0;
        color: #526579;
        font-size: 1.02rem;
    }

    .feature-card {
        padding: 1.1rem 1.2rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e3eaf2;
        min-height: 115px;
    }

    .feature-card h4 {
        margin: 0 0 0.35rem;
        color: #16324f;
    }

    .feature-card p {
        margin: 0;
        color: #657487;
        font-size: 0.92rem;
    }

    .disclaimer {
        padding: 0.9rem 1rem;
        border-radius: 12px;
        background: #fff8e6;
        border: 1px solid #f1dfaa;
        color: #6b5721;
        font-size: 0.88rem;
        margin-top: 1rem;
    }

    .source-box {
        padding: 0.8rem 1rem;
        border-radius: 10px;
        background: #f4f8fb;
        border: 1px solid #dfe8ef;
        margin-top: 0.6rem;
        font-size: 0.85rem;
    }

    .small-label {
        color: #718096;
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stChatMessage"] {
        border-radius: 16px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Models
# ---------------------------------------------------------

@st.cache_resource
def get_model():
    return ChatOpenRouter(
        model="z-ai/glm-5.3-flash",
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0,
    )


@st.cache_resource
def get_whisper():
    return WhisperModel(
        "small",
        device="cpu",
        compute_type="int8",
    )


model = get_model()
whisper_model = get_whisper()


# ---------------------------------------------------------
# Medical information tool
# ---------------------------------------------------------

def _clean_text(text):
    """Remove simple HTML tags returned by MedlinePlus."""
    return re.sub(r"<[^>]+>", "", text or "").strip()


@st.cache_data(ttl=3600)
def search_medical_information(topic: str) -> str:
    url = "https://wsearch.nlm.nih.gov/ws/query"
    params = {
        "db": "healthTopics",
        "term": topic,
        "retmax": 3,
        "rettype": "brief",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        root = ET.fromstring(response.text)
        results = []

        for document in root.findall(".//document"):
            title = ""
            summary = ""
            page_url = document.attrib.get("url", "")

            for content in document.findall("content"):
                name = content.attrib.get("name")
                text = _clean_text("".join(content.itertext()))

                if name == "title":
                    title = text
                elif name == "full-summary":
                    summary = text

            if title or summary:
                results.append(
                    {
                        "title": title,
                        "summary": summary,
                        "url": page_url,
                    }
                )

        if not results:
            return f"No medical information found for: {topic}"

        output = f"Medical information from MedlinePlus for '{topic}':\n\n"

        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"{result['summary']}\n"
            output += f"Source: {result['url']}\n\n"

        output += (
            "Important: This information is for educational purposes only. "
            "It does not provide a diagnosis or medical prescription."
        )

        return output

    except requests.RequestException as exc:
        return f"Unable to access MedlinePlus: {exc}"
    except ET.ParseError:
        return "Unable to process the medical information returned by MedlinePlus."


from langchain_core.tools import tool


@tool
def medical_information(topic: str) -> str:
    """
    Search MedlinePlus for general medical information about a topic.
    Provides educational information only and does not diagnose or prescribe.
    """
    return search_medical_information(topic)


# ---------------------------------------------------------
# Agent logic
# ---------------------------------------------------------

def get_answer(user_text):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    model_with_tools = model.bind_tools([medical_information])
    response = model_with_tools.invoke(messages)

    sources = []
    tool_used = False

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:
            if tool_call["name"] == "medical_information":
                tool_used = True
                tool_result = medical_information.invoke(tool_call["args"])

                for line in str(tool_result).splitlines():
                    if line.startswith("Source:"):
                        url = line.replace("Source:", "", 1).strip()
                        if url:
                            sources.append(url)

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        final_response = model.invoke(messages)
        answer = final_response.content

        if not answer:
            answer = (
                "I could not generate a response right now. "
                "Please try asking the question again."
            )

        return answer, sources, tool_used

    return response.content, sources, tool_used


# ---------------------------------------------------------
# Voice functions
# ---------------------------------------------------------

def transcribe_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp:
        temp.write(audio_bytes)
        temp_path = temp.name

    try:
        segments, _ = whisper_model.transcribe(temp_path)
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


async def generate_voice(text):
    output_path = tempfile.NamedTemporaryFile(
        suffix=".mp3",
        delete=False,
    ).name

    communicate = edge_tts.Communicate(
        text=text,
        voice="en-IN-NeerjaNeural",
    )

    await communicate.save(output_path)
    return output_path


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sources" not in st.session_state:
    st.session_state.sources = {}

if "audio_files" not in st.session_state:
    st.session_state.audio_files = {}

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 MedAssist")
    st.caption("Medical Information Voice Assistant")

    st.divider()

    st.markdown("### How it works")

    st.markdown(
        """
        **1. Ask a question**  
        Type or speak your medical question.

        **2. Speech-to-text**  
        Faster-Whisper converts your voice into text.

        **3. Medical search**  
        The assistant retrieves educational information from MedlinePlus.

        **4. AI response**  
        GLM-5.3-Flash summarizes the retrieved information.

        **5. Voice response**  
        Edge-TTS converts the answer into speech.
        """
    )

    st.divider()

    st.markdown("### Technologies")

    st.markdown(
        """
        - OpenRouter
        - GLM-5.3-Flash
        - LangChain
        - MedlinePlus
        - Faster-Whisper
        - Edge-TTS
        - Streamlit
        """
    )

    st.divider()

    if st.button("🧹 Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.sources = {}
        st.session_state.audio_files = {}
        st.rerun()

    st.markdown(
        """
        <div class="disclaimer">
        <b>Medical disclaimer</b><br>
        This application provides general educational information only.
        It does not diagnose conditions, prescribe medicines, or replace
        professional medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# Main page
# ---------------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🩺 MedAssist</h1>
        <p>
            Ask medical questions using text or your voice and receive
            concise, educational information supported by MedlinePlus.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🎤 Voice Input</h4>
            <p>Speak naturally and let Faster-Whisper convert your question into text.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="feature-card">
            <h4>📚 Trusted Information</h4>
            <p>Retrieve general health information from the MedlinePlus health topics database.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="feature-card">
            <h4>🔊 Voice Response</h4>
            <p>Listen to the assistant's concise answer using Edge-TTS.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------

for index, message in enumerate(st.session_state.messages):
    role = message["role"]
    content = message["content"]

    with st.chat_message(role):
        st.write(content)

        if role == "assistant":
            audio_path = st.session_state.audio_files.get(index)
            if audio_path and os.path.exists(audio_path):
                st.audio(audio_path, format="audio/mp3")

            source_list = st.session_state.sources.get(index, [])
            if source_list:
                with st.expander("📚 Sources"):
                    for source in source_list:
                        st.markdown(source)

# ---------------------------------------------------------
# Voice input
# ---------------------------------------------------------

st.markdown('<div class="small-label">Voice question</div>', unsafe_allow_html=True)

audio_input = st.audio_input(
    "🎤 Record your medical question",
    key="voice_input",
)

voice_question = None

if audio_input is not None:
    with st.spinner("Transcribing your question..."):
        try:
            voice_question = transcribe_audio(audio_input.getvalue())
        except Exception as exc:
            st.error(f"Voice transcription failed: {exc}")

    if voice_question:
        st.info(f"Transcribed question: {voice_question}")

# ---------------------------------------------------------
# Text input
# ---------------------------------------------------------

typed_question = st.chat_input(
    "💬 Type your medical question..."
)

user_question = typed_question or voice_question

# ---------------------------------------------------------
# Process question
# ---------------------------------------------------------

if user_question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_question,
        }
    )

    with st.chat_message("user"):
        st.write(user_question)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical information and preparing your answer..."):
            try:
                answer, sources, tool_used = get_answer(user_question)

                st.write(answer)

                # Generate voice response
                with st.spinner("Generating voice response..."):
                    audio_path = asyncio.run(generate_voice(answer))

                st.audio(audio_path, format="audio/mp3")

                assistant_index = len(st.session_state.messages)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                st.session_state.sources[assistant_index] = sources
                st.session_state.audio_files[assistant_index] = audio_path

                if sources:
                    with st.expander("📚 Sources used"):
                        for source in sources:
                            st.markdown(source)

            except Exception as exc:
                error_message = (
                    "I couldn't process that question right now. "
                    "Please try again in a moment."
                )
                st.error(error_message)
                st.caption(f"Technical detail: {exc}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------

st.markdown(
    """
    <div style="text-align:center; margin-top:2rem; color:#7a8794; font-size:0.8rem;">
        MedAssist • General educational information only • Not a medical diagnosis
    </div>
    """,
    unsafe_allow_html=True,
)
