import os
from pathlib import Path
from google import genai
from google.genai import types

# Automatically looks for the GEMINI_API_KEY environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Configure Gemini to achieve the most deterministic, reproducible, and reliable output possible

# -> Use system instructions instead of the prompt to define transcription rules
SYSTEM_INSTRUCTION = """
You are a verbatim audio transcription engine for two-speaker dialogue.
Your sole task is to transcribe dialogue into text verbatim. 
NEVER guess or hallucinate content. NEVER paraphrase, 
correct grammar, smooth over speech disfluencies, or add any commentary.
Output the transcript ONLY, one line per speaker turn, each turn on a single line, in this exact format:

Follow these strict formatting rules for the transcript:

<formatting_rules>
1. Format the transcript strictly as a clear dialog using [MM:SS] timestamps and speaker tags (Speaker A, Speaker B).
2. Every single line must follow this exact pattern: [MM:SS] Speaker X: [Speech text]
3. Do not include introductory text, concluding remarks, or markdown code blocks (like ```). Output ONLY the raw text transcript.
</formatting_rules>

Follow these strict rules when you are not certain about the clarity of the audio:

<handling_uncertainty>
1. If a word or phrase is entirely unintelligible, replace it exactly with the token: [unclear]
2. Do not attempt to guess, hallucinate, or fill in unclear parts based on context. 
3. If an entire line or timestamp is fuzzy, still output the timestamp and tag, using [unclear] for the text.
</handling_uncertainty>

<example_format>
[00:05] Speaker A: What can I get started for you today?
[00:08] Speaker B: Um, hello. I would like to know the price of a coffee milk.
[00:12] Speaker A: Sure thing.
</example_format>
"""

# -> Minimal instructions in the user prompt
USER_PROMPT = "Transcribe dialogue in the audio file verbatim, striclty following the system instructions."

# -> Prepare a config object for deterministic generation
CONFIG = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,        # all transcription rules, applied to every request
    temperature=0.0,                              # greedy decoding: always pick the most probable token
    top_k=1,                                      # only ever consider the single top token (explicit greedy)
    top_p=1.0,                                    # no nucleus filtering (inert at temp 0, set for clarity)
    seed=42,                                      # fixed seed; only has effect if temperature > 0
    candidate_count=1,                            # generate one response, not several
    max_output_tokens=32768,                      # generous headroom
    response_mime_type=None,                      # here its better not to constrain the output 
    response_schema=None,      # constrain output to a list of TranscriptSegment objects
    audio_timestamp=True,                         # enable real timestamp understanding for audio-only input
    thinking_config=types.ThinkingConfig(
        thinking_budget=0,                        # disable variable-length reasoning for stable output
    ),
)

# -> Choose a model with specific weights (the last three digits)
MODEL_NAME = "gemini-2.5-pro-002"  # Note: Pro is highly recommended over Flash for strict adherence to formatting rules

# Define the data directory
data_dir = Path("data")

# Audio files to look for in each folder
target_files = ["transcript_training.wav", "transcript_experiment.wav"]

# Loop through all folders in the data directory
for folder in data_dir.iterdir():
    if not folder.is_dir():
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing folder: {folder.name}")
    print(f"{'='*60}")
    
    # Look for target audio files in this folder
    for target_file in target_files:
        audio_path = folder / target_file

        if not audio_path.exists():
            print(f"  ⚠️  {target_file} not found in {folder.name}")
            continue

        # Determine where the transcript would be saved and skip if it already exists
        output_filename = target_file.replace(".wav", "_transcript.txt")
        output_path = folder / output_filename
        # if output_path.exists():
        #     print(f"  ℹ️  Transcript already exists at {output_path}, skipping {target_file}.")
        #     continue

        print(f"\n  Uploading {target_file} to Google's Files API...")
        audio_file = client.files.upload(file=str(audio_path))
        
        print(f"  Processing audio and formatting transcript...")


        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[USER_PROMPT, audio_file],
            config=CONFIG
        )
        
        # Output the result
        print(f"\n  --- Transcript for {target_file} ---")
        print(response.text)
        
        # Save to file in the same folder (output_path computed earlier)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(response.text)
        print(f"  ✓ Transcript saved to {output_path}")

print(f"\n{'='*60}")
print("All folders processed!")
print(f"{'='*60}")