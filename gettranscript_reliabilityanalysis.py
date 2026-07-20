import json
from pathlib import Path

from my_gemini_API_calls import transcribe_gemini
from my_asr_API_calls import diarize_audio, transcribe_qwen

# ----------------- GENERATE MULTIPLE TRANSCRIPT TAKES FOR RELIABILITY ANALYSIS -----------------

# Runs gettranscript N_TAKES times for each temperature in TEMPERATURES (Gemini) and once with
# pyannote+Qwen ASR, saving the results. The output files are named
# transcript_experiment_temp{temp}_take{take}.json / transcript_experiment_asr.json
# and are found by analysis_helpers.find_transcript_files() to compute reliability metrics.

TEMPERATURES = [0, 1.0]
N_TAKES = 5
DATA_DIR = Path("data")
PARTICIPANT_DIR = DATA_DIR / "experiment_Gerard_1780157613344"
TARGET_FILES = ["transcript_experiment.wav"]


def _save_take(target_file, suffix, transcribe_fn, redo=False):
    audio_path = PARTICIPANT_DIR / target_file
    if not audio_path.exists():
        return
    output_path = PARTICIPANT_DIR / "analysis" / target_file.replace(".wav", f"_{suffix}.json")
    if output_path.exists() and not redo:
        print(f"  ℹ️  Skipping {output_path} (already exists)")
        return
    print(f"\nTranscribing {audio_path}  suffix={suffix}")
    segments = transcribe_fn(audio_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Saved to {output_path}")


def run_transcriptions_gemini(redo=False):
    for temp in TEMPERATURES:
        temp_tag = str(int(temp))
        for take in range(1, N_TAKES + 1):
            suffix = f"temp{temp_tag}_take{take}"
            for target_file in TARGET_FILES:
                _save_take(target_file, suffix, lambda p: transcribe_gemini(p, temperature=temp),redo=redo)


def run_transcriptions_asr(redo=False):
    for take in range(1, N_TAKES + 1):
        suffix = f"asr_take{take}"
        for target_file in TARGET_FILES:
            _save_take(target_file, suffix, lambda p: transcribe_qwen(diarize_audio(str(p)), str(p)),redo=redo)


if __name__ == "__main__":
    run_transcriptions_gemini()
    run_transcriptions_asr(redo=True)  # redo=True to overwrite existing ASR transcripts, since they were generated with a different Qwen model
