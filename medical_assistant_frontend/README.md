# Medical Information Assistant

A Streamlit frontend for the notebook-based medical information assistant.

## Features

- Text questions
- Microphone input
- Faster-Whisper speech-to-text
- MedlinePlus information retrieval
- OpenRouter + GLM-5.3-Flash response generation
- Edge-TTS spoken responses
- Chat-style frontend

## Local setup

### 1. Create a virtual environment

```powershell
python -m venv venv
.env\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Create `.env`

Copy `.env.example` to `.env` and add your OpenRouter API key.

### 4. Start the app

```powershell
streamlit run app.py
```

The browser should open automatically.

## Important

This application is for general educational information only. It is not a diagnostic or prescription system. Do not use it as a substitute for professional medical care.

## Deployment

For a simple deployment, Streamlit Community Cloud can run this project from GitHub. Add `OPENROUTER_API_KEY` as a Streamlit secret instead of committing `.env`.

Because Faster-Whisper downloads a speech model and performs CPU inference, the first startup can be slow and deployment resources matter.
