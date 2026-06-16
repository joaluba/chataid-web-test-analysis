import os
import json
import time
from pathlib import Path
from typing import List, Dict
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
import os


api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

MODEL = "gemini-2.5-pro"  # Recommended over flash models for strict scoring accuracy and logical counting
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
# 2. Enhanced Core Functionality
# =====================================================================

def count_clarifications(transcript_text: str) -> dict:
    """
    Determines how many times a participant asked for clarification,
    utilizing structured outputs for absolute reproducibility.
    """
    system_instruction = (
        "You are an objective, analytical research assistant trained to identify and isolate "
        "conversational clarification requests in audio transcripts with complete precision. "
        "Do not guess or apply subjective interpretations; adhere strictly to the definitions provided."
    )

    prompt = f"""You are analyzing a conversation transcript.

Count the number of CLARIFICATION REQUESTS made by the participant (the tourist/customer, NOT the attendant/waiter).

A clarification request is strictly defined as an utterance where the participant asks for something to be repeated or re-explained because they did not hear or understand it.
Examples:
- "Sorry, which was the last museum?"
- "What was the artist's name again?"
- "Can you repeat that?"
- "Sorry?"
- "What?"
- "Can you speak louder?"

Do NOT count general follow-up questions that probe for new, unmentioned information (e.g., asking "Is there vegan milk?" when milk options haven't been discussed yet is NOT a clarification request).

TRANSCRIPT TO ANALYZE:
---
{transcript_text}
---
"""

    # Enforce deterministic constraints + JSON Schema targeting the Pydantic model
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        temperature=0.0,
        seed=42,
        candidate_count=1,
        response_mime_type="application/json",
        response_schema=ClarificationAnalysis,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=config
    )
    
    # Parse output reliably (Gemini SDK guarantees this structure matches ClarificationAnalysis)
    data = json.loads(response.text)
    return data


def score_answers(user_input_dict: dict, reference_answers: str, task_label: str) -> dict:
    """
    Scores participant responses objectively using rigorous semantic alignment
    and returning structured score mapping schemas.
    """
    system_instruction = (
        "You are an impartial academic grader scoring a user-input questionnaire against "
        "an official answer key. You score with absolute objectivity, consistency, and precision."
    )

    prompt = f"""You are scoring a participant's answers in a {task_label} task.

Below you will find the correct REFERENCE ANSWERS, followed by the participant's actual answers.
For each question key provided in the participant's answers, evaluate their accuracy and assign a score:
- 1.0 (Correct): The meaning is fully correct, even if not perfectly worded. For numerical responses, ignore differences in currency signs/units if the correct numerical value is correct.
- 0.5 (Partially Correct): Minor omissions or slight inaccuracies (e.g., missing one of several requested items, single-word errors).
- 0.0 (Incorrect): The answer is incorrect, irrelevant, or missing.

Ensure you match questions using semantic meaning, not string syntax. You must evaluate every key inside the participant's answers dictionary and map it strictly inside the output 'scores' object.

REFERENCE ANSWERS:
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
        response_mime_type="application/json",
        response_schema=ScoreMapping,
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=[prompt],
        config=config
    )

    data = json.loads(response.text)
    # Return just the dictionary of key-value score pairs to align with original script flow
    return data.get("scores", {})


# =====================================================================
# 3. Folder Pipeline Execution
# =====================================================================

def process_folder(folder: Path):
    print(f"\n{'='*60}\nProcessing: {folder.name}\n{'='*60}")

    train_tx = (folder / "transcript_training_transcript.txt").read_text(encoding="utf-8")
    exp_tx = (folder / "transcript_experiment_transcript.txt").read_text(encoding="utf-8")
    data = json.loads((folder / "experiment_data.json").read_text(encoding="utf-8"))

    # 1 & 2: clarification counts (strictly parsed out of standard schemas)
    train_clar = count_clarifications(train_tx)
    exp_clar = count_clarifications(exp_tx)
    print(f"  training clarifications:   {train_clar['clarification_requests']}")
    print(f"  experiment clarifications: {exp_clar['clarification_requests']}")

    # 3 & 4: answer scoring
    train_scores = score_answers(
        data.get("training_userinput", {}), TRAINING_ANSWERS, "training (tourist office)")
    exp_scores = score_answers(
        data.get("experiment_userinput", {}), EXPERIMENT_ANSWERS, "experiment (cafe)")
    print(f"  training score total:   {sum(train_scores.values())} / {len(train_scores)}")
    print(f"  experiment score total: {sum(exp_scores.values())} / {len(exp_scores)}")

    # 5: assemble enriched output
    enriched = dict(data)  # Keep original data
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