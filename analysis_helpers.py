import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import jiwer
import numpy as np

TARGET_FILES = ["transcript_experiment.wav"]
TEMPERATURES = [0, 1.0]
N_TAKES = 5


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


def analyseWER(participant_dir, target_files=TARGET_FILES, temperatures=TEMPERATURES, n_takes=N_TAKES):
    """
    Load pre-generated transcript takes from participant_dir/analysis/ and compute
    pairwise WER, summary statistics, and turn counts per speaker.

    Returns:
        {
          target_file: {
            int(temp): {
              "pairwise_wer":      [{"take_i", "take_j", "wer"}, ...],
              "mean_wer":          float,
              "median_wer":        float,
              "max_wer":           float,
              "turns_per_speaker": [{"take", "Speaker A", "Speaker B", ...}, ...],
            }
          }
        }
    """
    results = {}
    for target_file in target_files:
        results[target_file] = {}
        for temp in temperatures:
            temp_tag = f"temp{str(int(temp))}"

            transcripts = []
            for take in range(1, n_takes + 1):
                path = participant_dir / "analysis" / target_file.replace(
                    ".wav", f"_{temp_tag}_take{take}.json"
                )
                if path.exists():
                    transcripts.append(json.loads(path.read_text(encoding="utf-8")))

            label = f"{participant_dir.name} / {target_file} / temperature={temp}"
            if len(transcripts) < 2:
                print(f"\n[SKIP] {label}: only {len(transcripts)} transcript(s) found")
                continue

            print(f"\n{'='*60}")
            print(label)
            print(f"{'='*60}")

            texts = [segments_to_text(t) for t in transcripts]

            # Pairwise WER
            pairwise = []
            print("\nPairwise WER:")
            for i, j in combinations(range(len(transcripts)), 2):
                score = jiwer.wer(texts[i], texts[j])
                pairwise.append({"take_i": i + 1, "take_j": j + 1, "wer": score})
                print(f"  take{i+1} vs take{j+1}: {score:.4f}")

            wer_values = [p["wer"] for p in pairwise]
            print(f"\nMean WER:   {np.mean(wer_values):.4f}")
            print(f"Median WER: {np.median(wer_values):.4f}")
            print(f"Max WER:    {np.max(wer_values):.4f}")

            # Turns per speaker
            turns = []
            print("\nTurns per speaker:")
            for take_idx, segments in enumerate(transcripts):
                counts = dict(Counter(seg["speakerID"] for seg in segments))
                turns.append({"take": take_idx + 1, **counts})
                counts_str = "  ".join(f"{spk}: {n}" for spk, n in sorted(counts.items()))
                print(f"  take{take_idx+1}: {counts_str}")

            results[target_file][int(temp)] = {
                "pairwise_wer": pairwise,
                "mean_wer": float(np.mean(wer_values)),
                "median_wer": float(np.median(wer_values)),
                "max_wer": float(np.max(wer_values)),
                "turns_per_speaker": turns,
            }

    return results
