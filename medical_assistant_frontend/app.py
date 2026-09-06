import os
import tempfile
import xml.etree.ElementTree as ET

import edge_tts
import requests
import streamlit as st
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter


# -----------------------------
# Configuration
# -----------------------------
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
    page_title="Medical Information Assistant",
    page_icon="🩺",
    layout="centered",
)

SYSTEM_PROMPT = """
You are a medical information voice assistant.

Your job is to provide general educational medical information.

Important rules:
- Do not diagnose diseases.
- Do not prescribe medicines.
- Do not replace a healthcare professional.
- Use information retrieved from the medical_information tool.
- Keep the final response concise.
- Use simple language.
- Do not use Markdown in the spoken answer.
- Avoid long lists.
- If the user describes an emergency or severe symptoms, advise them to seek
  urgent professional medical care rather than attempting a diagnosis.
"""

# -----------------------------
# Models
# -----------------------------
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


# -----------------------------
# Medical information tool
# -----------------------------
def medical_information(topic: str) -> str:
    """Search MedlinePlus for general educational medical information."""
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
                text = "".join(content.itertext()).strip()

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


# -----------------------------
# Agent / tool loop
# -----------------------------
def get_answer(user_text: str):
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_text),
    ]

    response = model.bind_tools([medical_information]).invoke(messages)

    if response.tool_calls:
        messages.append(response)

        for tool_call in response.tool_calls:
            if tool_call["name"] == "medical_information":
                tool_result = medical_information(**tool_call["args"])

                messages.append(
                    ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    )
                )

        final_response = model.invoke(messages)
        answer = final_response.content
        return answer, messages, final_response

    return response.content, messages, response


# -----------------------------
# Text-to-speech
# -----------------------------
async def generate_voice(text: str, output_path: str):
    communicate = edge_tts.Communicate(
        text=text,
        voice="en-IN-NeerjaNeural",
    )
    await communicate.save(output_path)


# -----------------------------
# Speech-to-text
# -----------------------------
def transcribe_audio(uploaded_audio):
    suffix = ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_audio.getvalue())
        temp_path = tmp.name

    try:
        segments, _ = whisper_model.transcribe(
            temp_path,
            beam_size=5,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return text
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass


# -----------------------------
# UI
# -----------------------------
st.title("🩺 Medical Information Assistant")
st.caption("General educational information — not a diagnosis or prescription.")

with st.sidebar:
    st.header("About")
    st.write(
        "Ask a health-related question by typing or using your microphone. "
        "The assistant retrieves general information from MedlinePlus and "
        "uses an AI model to explain it in simple language."
    )
    st.warning(
        "For emergencies or severe symptoms, contact local emergency services "
        "or a qualified healthcare professional."
    )

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message.get("audio"):
            st.audio(message["audio"], format="audio/mp3")

st.subheader("Ask your question")

audio_input = st.audio_input("🎤 Record your question")

typed_question = st.chat_input(
    "Example: What are common symptoms of hypertension?"
)

user_text = None

if audio_input is not None:
    with st.spinner("Converting your voice to text..."):
        user_text = transcribe_audio(audio_input)

    if user_text:
        st.info(f"Transcribed question: {user_text}")
    else:
        st.warning("I could not understand the recording.")

elif typed_question:
    user_text = typed_question.strip()

if user_text:
    st.session_state.messages.append(
        {"role": "user", "content": user_text}
    )

    with st.chat_message("user"):
        st.write(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Searching medical information and generating answer..."):
            try:
                answer, _, _ = get_answer(user_text)

                if isinstance(answer, list):
                    answer = " ".join(
                        item.get("text", "")
                        for item in answer
                        if isinstance(item, dict)
                    ).strip()

                answer = str(answer).strip()

                if not answer:
                    answer = (
                        "I could not generate a response. "
                        "Please try asking the question again."
                    )

                st.write(answer)

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".mp3"
                ) as audio_file:
                    audio_path = audio_file.name

                import asyncio
                asyncio.run(generate_voice(answer, audio_path))

                st.audio(audio_path, format="audio/mp3")

                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "audio": audio_bytes,
                    }
                )

                try:
                    os.remove(audio_path)
                except OSError:
                    pass

            except Exception as exc:
                st.error(f"Something went wrong: {exc}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Sorry, I could not process that request.",
                    }
                )
