import json
from pathlib import Path

from my_gemini_API_calls import transcribe_audio

data_dir = Path("data")
target_files = ["transcript_training.wav", "transcript_experiment.wav"]

for folder in data_dir.iterdir():
    if not folder.is_dir():
        continue

    print(f"\n{'='*60}")
    print(f"Processing folder: {folder.name}")
    print(f"{'='*60}")

    for target_file in target_files:
        audio_path = folder / target_file

        if not audio_path.exists():
            print(f"  ⚠️  {target_file} not found in {folder.name}")
            continue

        output_filename = target_file.replace(".wav", ".json")
        output_path = folder / "analysis" / output_filename
        if output_path.exists():
            print(f"  ℹ️  Transcript already exists at {output_path}, skipping.")
            continue

        print(f"\n  Uploading {target_file} to Google's Files API...")
        segments = transcribe_audio(audio_path)

        print(f"\n  --- Transcript for {target_file} ({len(segments)} turns) ---")
        for seg in segments:
            print(f"  [{seg['timestamp']}] {seg['speakerID']}: {seg['text']}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  ✓ Transcript saved to {output_path}")

print(f"\n{'='*60}")
print("All folders processed!")
print(f"{'='*60}")
