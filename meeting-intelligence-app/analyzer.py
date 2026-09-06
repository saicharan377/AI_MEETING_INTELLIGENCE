import os
import time
import streamlit as st
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.errors import ServerError, APIError

load_dotenv()

# Read API key from Streamlit Cloud Secrets or local .env
api_key = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

if not api_key:
    raise ValueError("Missing GOOGLE_API_KEY. Configure it in Streamlit Cloud Secrets or your .env file.")

client = genai.Client(api_key=api_key)

# --- Structured Output Schema ---

class ActionItem(BaseModel):
    task: str = Field(description="Actionable task or deliverable")
    owner: Optional[str] = Field(default="Unassigned", description="Person or team responsible")
    due_date: Optional[str] = Field(default="Not specified", description="Deadline or timeline")

class MeetingReport(BaseModel):
    meeting_title: str = Field(description="Clear title for the discussion")
    executive_summary: List[str] = Field(description="3-5 concise bullet points capturing core outcomes")
    key_decisions: List[str] = Field(description="Decisions agreed upon in the meeting")
    action_items: List[ActionItem] = Field(description="List of extracted actionable items")

# Fallback models in priority order
MODELS_TO_TRY = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash-latest"]

def analyze_transcript(transcript_text: str) -> MeetingReport:
    """Extracts summary and actions with automatic fallback across models."""
    prompt = (
        "You are an executive assistant. Extract decisions, summaries, and action items "
        f"from this meeting transcript:\n\n{transcript_text}"
    )

    last_exception = None

    for model_name in MODELS_TO_TRY:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": MeetingReport,
                        "temperature": 0.1,
                    },
                )
                return MeetingReport.model_validate_json(response.text)
            except (ServerError, APIError) as e:
                last_exception = e
                time.sleep(2)
                continue

    raise RuntimeError(f"Gemini API temporarily unavailable across all endpoints: {last_exception}")

def ask_meeting_chat(transcript_text: str, chat_history: list, user_question: str) -> str:
    """Answers user questions strictly grounded in the meeting transcript."""
    system_instruction = (
        "You are an assistant answering questions about a specific meeting. "
        "Use ONLY the provided meeting transcript to answer questions. "
        "If the answer is not mentioned or cannot be inferred directly, say so clearly.\n\n"
        f"Meeting Transcript:\n{transcript_text}"
    )

    contents = []
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)]
        )
    )

    for model_name in MODELS_TO_TRY:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            return response.text
        except (ServerError, APIError):
            time.sleep(1)
            continue

    return "The model service is temporarily busy. Please try asking again in a moment."