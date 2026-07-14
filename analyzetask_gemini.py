import os
import json
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from google.genai.types import ThinkingConfig


api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL = "gemini-2.5-pro-001"
DATA_DIR = Path("data")

# Reference answers (constant across participants)
TRAINING_ANSWERS = """
- Price of the single metro ticket: 2.5 euros
- Temperature of the water in the sea: 15 degrees
- Museums with free entrance today: Museum of Pablo Picasso, Museum of Design, Museum of Science
- Tourist office closing time: 10pm
"""

EXPERIMENT_ANSWERS = """
- Price of coffee with milk: 1.9 euros
- Available milk options: Almond, Coconut, Cow Milk
- Is there extra charge for vegan milk: Yes, 20 cents
- Specialty cake: Tarta de Santiago (almond cake)
- Wifi Name: Coffee And Jazz (written together)
- Wifi Password: Enjoy Your Coffee (written together)
- Maximum table duration: 90 minutes
- Today's event: Jazz concert
- Name of the artist: Barcelona Jazz Collective
- Cafe closing time: 2am
"""

VALID_SCORES = {0.0, 0.5, 1.0}


# =====================================================================
# 1. Pydantic Schemas for Structured JSON Generation
# =====================================================================

class ClarificationAnalysis(BaseModel):
    clarification_requests: int = Field(
        description="The exact count of clarification requests made ONLY by the participant (the tourist/customer)."
    )
    examples: List[str] = Field(
        description="Strict, literal quotes of each clarification request made by the participant."
    )


class ScoreMapping(BaseModel):
    scores: Dict[str, float] = Field(
        description="An explicit mapping of each exact question key to its numeric score: 1.0 (correct), 0.5 (partially correct), or 0.0 (incorrect)."
    )


# =====================================================================
# 2. Helpers
# =====================================================================

def clamp_score(v: float) -> float:
    return min(VALID_SCORES, key=lambda s: abs(s - v))


# =====================================================================
# 3. Core Functionality
# =====================================================================

def count_clarifications(transcript_text: str) -> dict:
    system_instruction = (
        "You are an objective, analytical research assistant trained to identify and isolate "
        "conversational clarification requests in audio transcripts with complete precision. "
        "Do not guess or apply subjective interpretations; adhere strictly to the definitions provided.\n\n"
        "TASK: Count the number of CLARIFICATION REQUESTS made by the participant "
        "(the tourist/customer, NOT the attendant/waiter).\n\n"
        "A clarification request is strictly defined as an utterance where the participant asks "
        "for something to be repeated or re-explained because they did not hear or understand it.\n"
        "Examples of valid clarification requests:\n"
        "- 'Sorry, which was the last museum?'\n"
        "- 'What was the artist's name again?'\n"
        "- 'Can you repeat that?'\n"
        "- 'Sorry?'\n"
        "- 'What?'\n"
        "- 'Can you speak louder?'\n\n"
        "Do NOT count general follow-up questions that probe for new, unmentioned information "
        "(e.g., asking 'Is there vegan milk?' when milk options have not been discussed yet "
        "is NOT a clarification request).\n"
        "Only count utterances from the participant (tourist/customer), never from the attendant or waiter."
    )

    prompt = f"""TRANSCRIPT TO ANALYZE:
---
{transcript_text}
---
"""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        seed=42,
        candidate_count=1,
        thinking_config=ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=ClarificationAnalysis,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=config
    )

    data = json.loads(response.text)
    return data


def score_answers(user_input_dict: dict, reference_answers: str, task_label: str) -> dict:
    system_instruction = (
        "You are an impartial academic grader scoring a user-input questionnaire against "
        "an official answer key. You score with absolute objectivity, consistency, and precision.\n\n"
        f"TASK: Score a participant's answers in a {task_label} task.\n\n"
        "For each question key provided in the participant's answers, evaluate accuracy and assign exactly one of these scores:\n"
        "- 1.0 (Correct): The meaning is fully correct, even if not perfectly worded. "
        "For numerical responses, ignore differences in currency signs or units if the correct numerical value is present.\n"
        "- 0.5 (Partially Correct): Minor omissions or slight inaccuracies "
        "(e.g., missing one of several requested items, single-word errors).\n"
        "- 0.0 (Incorrect): The answer is incorrect, irrelevant, or missing.\n\n"
        "Match questions using semantic meaning, not string syntax. "
        "You MUST evaluate every key inside the participant's answers dictionary and map it strictly inside the output 'scores' object. "
        "No key may be omitted from the output."
    )

    prompt = f"""REFERENCE ANSWERS:
---
{reference_answers}
---

PARTICIPANT'S ANSWERS:
---
{json.dumps(user_input_dict, ensure_ascii=False, indent=2)}
---
"""

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        seed=42,
        candidate_count=1,
        thinking_config=ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
        response_schema=ScoreMapping,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=config
    )

    data = json.loads(response.text)
    raw_scores = data.get("scores", {})

    # Clamp all scores to {0.0, 0.5, 1.0}
    clamped = {k: clamp_score(v) for k, v in raw_scores.items()}

    # Validate key coverage
    input_keys = set(user_input_dict.keys())
    output_keys = set(clamped.keys())
    missing = input_keys - output_keys
    if missing:
        print(f"  WARNING: scorer did not return scores for keys: {missing}")

    return clamped


# =====================================================================
# 4. Folder Pipeline Execution
# =====================================================================

def process_folder(folder: Path):
    print(f"\n{'='*60}\nProcessing: {folder.name}\n{'='*60}")

    train_tx = (folder / "transcript_training_transcript.txt").read_text(encoding="utf-8")
    exp_tx = (folder / "transcript_experiment_transcript.txt").read_text(encoding="utf-8")
    data = json.loads((folder / "experiment_data.json").read_text(encoding="utf-8"))

    train_clar = count_clarifications(train_tx)
    exp_clar = count_clarifications(exp_tx)
    print(f"  training clarifications:   {train_clar['clarification_requests']}")
    print(f"  experiment clarifications: {exp_clar['clarification_requests']}")

    train_scores = score_answers(
        data.get("training_userinput", {}), TRAINING_ANSWERS, "training (tourist office)")
    exp_scores = score_answers(
        data.get("experiment_userinput", {}), EXPERIMENT_ANSWERS, "experiment (cafe)")
    print(f"  training score total:   {sum(train_scores.values())} / {len(train_scores)}")
    print(f"  experiment score total: {sum(exp_scores.values())} / {len(exp_scores)}")

    enriched = dict(data)
    enriched["analysis"] = {
        "clarification_requests_training": train_clar["clarification_requests"],
        "clarification_requests_experiment": exp_clar["clarification_requests"],
        "clarification_examples_training": train_clar.get("examples", []),
        "clarification_examples_experiment": exp_clar.get("examples", []),
        "training_userinput_scores": train_scores,
        "experiment_userinput_scores": exp_scores,
        "training_score_total": sum(train_scores.values()),
        "experiment_score_total": sum(exp_scores.values()),
    }

    out_path = folder / "experiment_data_plus_analysis.json"
    out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  ✓ saved {out_path}")


def main():
    if not DATA_DIR.exists():
        print(f"Error: '{DATA_DIR}' directory not found. Please place experiment folders in a './data' directory.")
        return

    for folder in DATA_DIR.iterdir():
        if folder.is_dir():
            # Uncomment if you want to skip folders that are already processed:
            # if (folder / "experiment_data_plus_analysis.json").exists():
            #     print(f"\n{'='*60}\nSkipping {folder.name} (already has analysis)\n{'='*60}")
            #     continue
            try:
                process_folder(folder)
            except Exception as e:
                print(f"  ✗ error in {folder.name}: {e}")

    print(f"\n{'='*60}\nDone.\n{'='*60}")


if __name__ == "__main__":
    main()
