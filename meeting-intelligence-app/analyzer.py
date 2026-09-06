import os
from typing import List, Optional
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

load_dotenv()

# --- Pydantic Schemas for Structured Output ---

class ActionItem(BaseModel):
    task: str = Field(description="Actionable task or deliverable")
    owner: Optional[str] = Field(default="Unassigned", description="Person or team responsible")
    due_date: Optional[str] = Field(default="Not specified", description="Deadline or timeline")

class MeetingReport(BaseModel):
    meeting_title: str = Field(description="Clear and concise title for the meeting discussion")
    executive_summary: List[str] = Field(description="3-5 concise bullet points capturing core outcomes")
    key_decisions: List[str] = Field(description="Decisions agreed upon in the meeting")
    action_items: List[ActionItem] = Field(description="List of extracted actionable items")

# --- Gemini Client Initialization ---

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- Core Intelligence Extraction ---

def analyze_transcript(transcript_text: str) -> MeetingReport:
    """
    Extracts structured executive summary, decisions, and action items
    from a timestamped meeting transcript using Gemini structured output.
    """
    prompt = (
        "You are an executive assistant. Extract decisions, summaries, and action items "
        f"from this meeting transcript:\n\n{transcript_text}"
    )
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_schema": MeetingReport,
            "temperature": 0.1,
        },
    )
    
    return MeetingReport.model_validate_json(response.text)

# --- Meeting Q&A Chatbot Function ---

def ask_meeting_chat(transcript_text: str, chat_history: list, user_question: str) -> str:
    """
    Answers user questions strictly grounded in the meeting transcript.
    Maintains conversational context from past messages in the session.
    """
    system_instruction = (
        "You are an assistant answering questions about a specific meeting. "
        "Use ONLY the provided meeting transcript to answer questions. "
        "If the answer is not mentioned or cannot be inferred directly from the transcript, "
        "say so clearly without hallucinating.\n\n"
        f"Meeting Transcript:\n{transcript_text}"
    )
    
    # Build the multi-turn contents list for the API
    contents = []
    
    # Pass prior chat history
    for msg in chat_history:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=msg["content"])]
            )
        )
    
    # Add current user question
    contents.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_question)]
        )
    )
    
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.2,
        ),
    )
    
    return response.text