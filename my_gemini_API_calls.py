import json
import os
from pathlib import Path

from google import genai
from google.genai import types

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))



# ── AUDIO TRANSCRIPTION ───────────────────────────────────────────────────────
#
# Reliability mechanisms used in this API call:
#
# 1. Pinned model version (MODEL_NAME)
#    Always calls the same model variant, preventing silent behaviour changes
#    when Gemini updates its default weights.
#
# 2. Low/zero temperature
#    Passed as a parameter so it can be varied experimentally, but defaults to
#    0.0 (greedy decoding) for maximum determinism.
#
# 3. Structured output (response_mime_type + response_schema)
#    Forces the model to emit valid JSON matching the schema, eliminating
#    free-form formatting variation and making output machine-parseable
#    without post-processing heuristics.
#
# 4. thinking_budget=0
#    Disables the model's internal chain-of-thought reasoning, which is itself
#    stochastic and can alter the final answer even at temperature=0.
#
# 5. System instruction (separate from user prompt)
#    Transcription rules are in the system turn rather than the user prompt,
#    giving them higher priority and keeping the user prompt minimal to reduce
#    instruction-following drift.
#
# ─────────────────────────────────────────────────────────────────────────────

MODEL_NAME = "gemini-2.5-flash"

TRANSCRIPTION_SYSTEM_INSTRUCTION = """
You are a verbatim audio transcription engine for two-speaker dialogue.
Your sole task is to transcribe the dialogue into structured JSON verbatim.
NEVER guess or hallucinate content. NEVER paraphrase,
correct grammar, smooth over speech disfluencies, or add any commentary.

Rules:
- One object per speaker turn.
- timestamp: start time of the turn in MM:SS format.
- speakerID: "Speaker A" for the agent/attendant, "Speaker B" for the user/tourist/customer.
- text: verbatim speech. Replace any unintelligible word or phrase with the token [unclear].
  Do NOT guess or fill in unclear parts from context.
"""

TRANSCRIPTION_USER_PROMPT = "Transcribe dialogue in the audio file verbatim, striclty following the system instructions."

TRANSCRIPTION_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "timestamp": {"type": "string"},
            "speakerID":  {"type": "string"},
            "text":       {"type": "string"},
        },
        "required": ["timestamp", "speakerID", "text"],
    },
}


def transcribe_gemini(audio_path: str | Path, temperature: float = 0.0) -> list[dict]:
    """Upload an audio file to Gemini and return the verbatim transcript as a list of segments."""
    config = types.GenerateContentConfig(
        system_instruction=TRANSCRIPTION_SYSTEM_INSTRUCTION,
        temperature=temperature,
        top_k=1,
        top_p=1.0,
        # seed=42, # at temperature=0, there is no randomness, so seeding is unnecessary
        candidate_count=1,
        max_output_tokens=8192,
        response_mime_type="application/json",
        response_schema=TRANSCRIPTION_SCHEMA,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )

    audio_file = client.files.upload(file=str(audio_path))
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[TRANSCRIPTION_USER_PROMPT, audio_file],
        config=config,
    )
    return json.loads(response.text)
