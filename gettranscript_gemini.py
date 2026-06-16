import os
from pathlib import Path
from google import genai
from google.genai import types

# Automatically looks for the GEMINI_API_KEY environment variable
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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

        # 1. Define strict system instructions to anchor the model's behavior
        system_instruction = (
            "You are a precise, literal, and objective audio transcription assistant. "
            "Your sole task is to transcribe audio into text verbatim. Do not paraphrase, "
            "correct grammar, smooth over speech disfluencies, or add any commentary."
        )

        # 2. Enforce deterministic generation configuration
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.0,       # Forces greedy decoding for maximum determinism
            seed=42,               # Anchors the random number generator
            candidate_count=1,     # Only generate the single best match
        )

        # 3. Construct the structured user prompt
        user_prompt = """
        You are transcribing a research interview audio file. Follow these rules with absolute precision:

        <formatting_rules>
        1. Format the transcript strictly as a clear dialog using [MM:SS] timestamps and speaker tags (Speaker A, Speaker B).
        2. Every single line must follow this exact pattern: [MM:SS] Speaker X: [Speech text]
        3. Do not include introductory text, concluding remarks, or markdown code blocks (like ```). Output ONLY the raw text transcript.
        </formatting_rules>

        <handling_uncertainty>
        1. If a word or phrase is entirely unintelligible, replace it exactly with the token: [unclear]
        2. Do not attempt to guess, hallucinate, or fill in unclear parts based on context. 
        3. If an entire line or timestamp is fuzzy, still output the timestamp and tag, using [unclear] for the text.
        </handling_uncertainty>

        <example_format>
        Speaker A: Hello, how are you?
        Speaker B: I'm good, thanks! And you?
        Speaker A: I wanted to ask about the [unclear] data you collected.
        </example_format>

        Transcribe the attached audio according to these rules.
        """

        response = client.models.generate_content(
            model="gemini-2.5-pro",  # Note: Pro is highly recommended over Flash for strict adherence to formatting rules
            contents=[user_prompt, audio_file],
            config=config
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