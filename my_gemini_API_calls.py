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


# ── CLARIFICATION REQUEST DETECTION ────────────────────────────────────────────
#
# Two variants of the same task, at different points in the pipeline:
#
# - propose_clarification_requests: permissive, over-inclusive candidate generation.
#   Meant to be reviewed and filtered by a human afterwards (see humanrate_NCR.py),
#   so it errs toward flagging borderline cases rather than missing real ones.
#
# - judge_clarification_requests: the final, no-human-in-the-loop decision. Used to
#   compare LLM-only judgments against the human-reviewed proposals for reliability
#   analysis, so it must apply the same definition precisely instead of over-including.
#
# ─────────────────────────────────────────────────────────────────────────────

CLARIFICATION_DEFINITION = (
    "A clarification request is an utterance where a speaker:\n"
    "- Asks to repeat or re-say something ('Can you repeat that?', 'Say that again?')\n"
    "- Asks for clarification of something already mentioned\n"
    "- Partially repeats something with a trailing or rising intonation\n"
    "- Asks for spelling, pronunciation, or confirmation of a word or phrase\n"
    "- Expresses that they didn't fully hear or understand something already said\n"
    "- Makes any request to speak louder or more clearly\n"
    "- Makes a sound indicating that they did not hear properly ('Huh?', 'Sorry?', 'What?')\n\n"
    "Do NOT include general information-seeking questions about new topics.\n"
    "Only mark utterances of the user (usually Speaker B) never the agent (Speaker A)."
)

CLARIFICATION_PROPOSAL_SYSTEM_INSTRUCTION = (
    "You are a linguistic expert identifying potential clarification requests in conversation "
    "transcripts, to be reviewed by a human afterwards. Mark ANY utterance from the user (usually "
    "Speaker B) (the participant/tourist/customer) that could possibly be a clarification request — "
    "err heavily on the side of inclusion.\n\n" + CLARIFICATION_DEFINITION
)

CLARIFICATION_FINAL_SYSTEM_INSTRUCTION = (
    "You are a linguistic expert making the FINAL determination of which utterances in a conversation "
    "transcript are genuine clarification requests. Your decision will not be reviewed by a human "
    "afterwards, so apply the definition precisely: mark an utterance ONLY if you are confident it is "
    "a genuine clarification request, not merely a possible one. Do not over-include borderline "
    "cases.\n\n" + CLARIFICATION_DEFINITION
)

CLARIFICATION_USER_PROMPT = (
    "Identify clarification requests in the transcript below according to the system instruction. "
    "Each line is prefixed with its index in brackets, e.g. [0], [1], etc. "
    "Return ONLY a JSON object with a single key \"cr_line_indices\" whose value is an array "
    "of the integer indices of lines that are clarification requests. "
    "Example: {\"cr_line_indices\": [2, 7, 12]}\n\n"
    "TRANSCRIPT:\n"
)


def _cr_line_indices(system_instruction: str, numbered_transcript: str) -> set[int]:
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        seed=42,
        candidate_count=1,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[f"{CLARIFICATION_USER_PROMPT}{numbered_transcript}"],
        config=config,
    )
    result = json.loads(response.text)
    return set(result.get("cr_line_indices", []))


def propose_clarification_requests(numbered_transcript: str) -> set[int]:
    """Ask Gemini which lines of a numbered transcript (one utterance per line, each
    prefixed "[i] ...") are POTENTIAL clarification requests from the user, erring
    toward over-inclusion since these proposals are meant for human review.
    Returns the set of flagged line indices."""
    return _cr_line_indices(CLARIFICATION_PROPOSAL_SYSTEM_INSTRUCTION, numbered_transcript)


def judge_clarification_requests(numbered_transcript: str) -> set[int]:
    """Ask Gemini for its FINAL, no-human-review judgment of which lines of a numbered
    transcript are genuine clarification requests from the user. Returns the set of
    flagged line indices."""
    return _cr_line_indices(CLARIFICATION_FINAL_SYSTEM_INSTRUCTION, numbered_transcript)
