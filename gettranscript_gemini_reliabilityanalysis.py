import json
from pathlib import Path

from my_gemini_API_calls import transcribe_audio

# ----------------- GENERATE MULTIPLE TRANSCRIPT TAKES FOR RELIABILITY ANALYSIS -----------------

# Runs gettranscript N_TAKES times for each temperature in TEMPERATURES and saves the results.
# The output files are named transcript_experiment_temp{temp}_take{take}.json
# and are used by analysis_helpers.analyseWER() to compute reliability metrics.

TEMPERATURES = [0, 1.0]
N_TAKES = 5
DATA_DIR = Path("data")
PARTICIPANT_DIR = DATA_DIR / "experiment_Julien_1779802077179"
TARGET_FILES = ["transcript_experiment.wav"]


def run_transcriptions():
    for temp in TEMPERATURES:
        temp_tag = str(int(temp))
        for take in range(1, N_TAKES + 1):
            suffix = f"temp{temp_tag}_take{take}"
            for target_file in TARGET_FILES:
                audio_path = PARTICIPANT_DIR / target_file
                if not audio_path.exists():
                    continue
                output_path = PARTICIPANT_DIR / "analysis" / target_file.replace(".wav", f"_{suffix}.json")
                if output_path.exists():
                    print(f"  ℹ️  Skipping {output_path} (already exists)")
                    continue
                print(f"\nTranscribing {audio_path}  suffix={suffix}  temperature={temp}")
                segments = transcribe_audio(audio_path, temperature=temp)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  ✓ Saved to {output_path}")


if __name__ == "__main__":
    run_transcriptions()
