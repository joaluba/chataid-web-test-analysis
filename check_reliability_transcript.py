import json
from collections import Counter
from itertools import combinations
from pathlib import Path
import jiwer
import numpy as np

from my_gemini_API_calls import transcribe_audio

# ----------------- ANALYSIS OF GEMINI TRANSCRIPTION RELIABILITY -----------------

# The goal of this script is to analyse the reliability of Gemini's verbatim transcription of a two-speaker dialogue.
# The script will generate multiple transcripts of the same audio file using different temperatures,
# and then compare the transcripts to see how consistent they are. Metrics that are used:
# - Word Error Rate (WER) between pairs of transcripts
# - Mean, median, and max WER across all pairs of transcripts
# - Number of turns per speaker in each transcript


TEMPERATURES = [0, 1.0]
N_TAKES = 5
DATA_DIR = Path("data")
PARTICIPANT_DIR = DATA_DIR / "experiment_Albert Barreiro_1779785968775"
TARGET_FILES = ["transcript_experiment.wav"]


# ── Part 1: Generate transcripts ─────────────────────────────────────────────

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
                # if output_path.exists():
                #     print(f"  ℹ️  Skipping {output_path} (already exists)")
                #     continue
                print(f"\nTranscribing {audio_path}  suffix={suffix}  temperature={temp}")
                segments = transcribe_audio(audio_path, temperature=temp)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"  ✓ Saved to {output_path}")


# ── Part 2: Analyse reliability ───────────────────────────────────────────────

def get_text(seg):
    if "text" in seg:
        return seg["text"]
    fallback_key = next((k for k in seg if k.startswith("text")), None)
    if fallback_key:
        print(f"  ⚠️  Schema violation: field is '{fallback_key}' instead of 'text'. Using its content as fallback.")
        return seg[fallback_key]
    raise KeyError(f"No 'text'-like key found in segment: {seg}")


def segments_to_text(segments):
    return " ".join(get_text(seg) for seg in segments)


def analyse():
    for target_file in TARGET_FILES:
        for temp in [0, 1]:
            temp_tag = f"temp{temp}"

            transcripts = []
            for take in range(1, N_TAKES + 1):
                path = PARTICIPANT_DIR / "analysis" / target_file.replace(
                    ".wav", f"_{temp_tag}_take{take}.json"
                )
                if path.exists():
                    transcripts.append(json.loads(path.read_text(encoding="utf-8")))

            label = f"{PARTICIPANT_DIR.name} / {target_file} / temperature={temp}"
            if len(transcripts) < 2:
                print(f"\n[SKIP] {label}: only {len(transcripts)} transcript(s) found")
                continue

            print(f"\n{'='*60}")
            print(label)
            print(f"{'='*60}")

            texts = [segments_to_text(t) for t in transcripts]

            # Pairwise WER
            wer_values = []
            print("\nPairwise WER:")
            for i, j in combinations(range(len(transcripts)), 2):
                score = jiwer.wer(texts[i], texts[j])
                wer_values.append(score)
                print(f"  take{i+1} vs take{j+1}: {score:.4f}")

            print(f"\nMean WER:   {np.mean(wer_values):.4f}")
            print(f"Median WER: {np.median(wer_values):.4f}")
            print(f"Max WER:    {np.max(wer_values):.4f}")

            # Turns per speaker
            print("\nTurns per speaker:")
            for take_idx, segments in enumerate(transcripts):
                counts = Counter(seg["speakerID"] for seg in segments)
                counts_str = "  ".join(f"{spk}: {n}" for spk, n in sorted(counts.items()))
                print(f"  take{take_idx+1}: {counts_str}")


if __name__ == "__main__":
    run_transcriptions()
    analyse()
