import os
import json
import time
import re
import streamlit as st
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()

# Streamlit Cloud secrets support with local .env fallback
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY in Streamlit Secrets or .env file.")

client = genai.Client(api_key=api_key)

class ActionItem(BaseModel):
    task: str = Field(description="Actionable task or deliverable")
    owner: Optional[str] = Field(default="Unassigned", description="Person or team responsible")
    due_date: Optional[str] = Field(default="Not specified", description="Deadline or timeline")

class MeetingReport(BaseModel):
    meeting_title: str = Field(description="Clear title for the discussion")
    executive_summary: List[str] = Field(description="3-5 concise bullet points capturing core outcomes")
    key_decisions: List[str] = Field(description="Decisions agreed upon in the meeting")
    action_items: List[ActionItem] = Field(description="List of extracted actionable items")

# Active current-generation models recommended by the Gemini API
MODELS = [
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3-flash-preview"
]

def analyze_transcript(transcript_text: str) -> MeetingReport:
    """Extracts summary and action items using standard prompt-driven JSON extraction."""
    system_prompt = (
        "You are an executive meeting assistant. Analyze the transcript and extract key outcomes. "
        "Return ONLY a valid, raw JSON object (no markdown tags, no ```json formatting) adhering to this schema:\n"
        "{\n"
        '  "meeting_title": "String",\n'
        '  "executive_summary": ["Point 1", "Point 2"],\n'
        '  "key_decisions": ["Decision 1"],\n'
        '  "action_items": [{"task": "Task description", "owner": "Name", "due_date": "Timeline"}]\n'
        "}"
    )

    last_err = None
    for model_name in MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user", 
                        parts=[types.Part.from_text(text=f"{system_prompt}\n\nTranscript:\n{transcript_text}")]
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            raw_text = response.text.strip()
            raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text)
            raw_text = re.sub(r"\s*```$", "", raw_text)
            
            data = json.loads(raw_text)
            return MeetingReport.model_validate(data)
        except Exception as e:
            last_err = e
            time.sleep(1)
            continue

    st.error(f"Gemini API Error Detail: {last_err}")
    raise RuntimeError(f"Gemini API Error: {last_err}")

def ask_meeting_chat(transcript_text: str, chat_history: list, user_question: str) -> str:
    """Answers user queries grounded directly in the meeting transcript."""
    system_instruction = (
        "You are an assistant answering questions about a specific meeting. "
        "Use ONLY the provided transcript. If the information is not mentioned, state that clearly.\n\n"
        f"Transcript:\n{transcript_text}"
    )

    contents = []
    for msg in chat_history:
        contents.append(types.Content(
            role="user" if msg["role"] == "user" else "model",
            parts=[types.Part.from_text(text=msg["content"])]
        ))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_question)]))

    for model_name in MODELS:
        try:
            res = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2
                )
            )
            return res.text
        except Exception:
            continue

    return "Unable to process request right now. Please try again."