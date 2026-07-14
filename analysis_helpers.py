import json
import re
import string
from itertools import combinations
from pathlib import Path

import jiwer
import numpy as np
from word2number.w2n import american_number_system, word_to_num

NUMBER_WORDS = set(american_number_system.keys())

NORMALIZE_TRANSFORM = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(),
])


def _normalize_numbers(text):
    """Rewrite spelled-out numbers ("twenty", "one point nine") as digits, so they
    compare equal to numerals ("20", "1.9") regardless of which form a transcript uses."""
    tokens = text.replace("-", " ").split()
    out = []
    buffer = []

    def flush():
        if not buffer:
            return
        try:
            out.append(str(word_to_num(" ".join(buffer))))
        except ValueError:
            out.extend(buffer)
        buffer.clear()

    def clean(tok):
        return tok.lower().strip(string.punctuation)

    for i, tok in enumerate(tokens):
        word = clean(tok)
        next_is_number = i + 1 < len(tokens) and clean(tokens[i + 1]) in NUMBER_WORDS
        if word in NUMBER_WORDS or (word == "and" and buffer and next_is_number):
            buffer.append(word)
        else:
            flush()
            out.append(tok)
    flush()

    return " ".join(out)


def _strip_bracketed_artifacts(text):
    """Remove bracketed ASR artifacts like "[music]" or "(inaudible)", including
    their contents, so they don't count as literal words when comparing transcripts."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_wer(reference, hypothesis, normalize_text=True):
    if normalize_text:
        return jiwer.wer(
            reference, hypothesis,
            reference_transform=NORMALIZE_TRANSFORM,
            hypothesis_transform=NORMALIZE_TRANSFORM,
        )
    return jiwer.wer(reference, hypothesis)


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

def find_transcript_files(participant_dir, tag):
    """Find every take file for one transcription method (e.g. "temp0", "temp1",
    "asr") in participant_dir/analysis/, sorted by take number."""
    return sorted(Path(participant_dir).glob(f"analysis/*_{tag}_take*.json"))

def load_transcripts(files):
    return [json.loads(Path(f).read_text(encoding="utf-8")) for f in files]

def _wer_stats(pairwise):
    wer_values = [p["wer"] for p in pairwise]
    return {
        "mean_wer": float(np.mean(wer_values)),
        "median_wer": float(np.median(wer_values)),
        "max_wer": float(np.max(wer_values)),
        "std_wer": float(np.std(wer_values)),
    }


def analyse_wer_within(files, label, normalize_text=True):
    """Pairwise WER between every take in `files` (repeated takes of a single
    transcription method).

    normalize_text: if True (default), lowercase and strip punctuation/extra
    whitespace before computing WER, so casing/punctuation differences don't
    count as errors.

    Returns None if fewer than 2 transcripts were found, otherwise:
        {
          "mean_wer":   float,
          "median_wer": float,
          "max_wer":    float,
          "std_wer":    float,
        }
    """
    transcripts = load_transcripts(files)
    if len(transcripts) < 2:
        print(f"\n[SKIP] {label}: only {len(transcripts)} transcript(s) found")
        return None

    texts = [segments_to_text(t) for t in transcripts]
    if normalize_text:
        texts = [_strip_bracketed_artifacts(t) for t in texts]
        texts = [_normalize_numbers(t) for t in texts]

    pairwise = [
        {"take_i": i + 1, "take_j": j + 1, "wer": compute_wer(texts[i], texts[j], normalize_text)}
        for i, j in combinations(range(len(transcripts)), 2)
    ]
    return _wer_stats(pairwise)


def analyse_wer_across(files_a, files_b, label, normalize_text=True):
    """Pairwise WER between every take in `files_a` and every take in `files_b`
    (e.g. Gemini takes vs ASR takes), as a measure of agreement between two
    transcription methods.

    Returns None if either list has no transcripts, otherwise the same shape
    as analyse_wer_within().
    """
    transcripts_a = load_transcripts(files_a)
    transcripts_b = load_transcripts(files_b)
    if not transcripts_a or not transcripts_b:
        print(f"\n[SKIP] {label}: {len(transcripts_a)} vs {len(transcripts_b)} transcript(s) found")
        return None

    texts_a = [segments_to_text(t) for t in transcripts_a]
    texts_b = [segments_to_text(t) for t in transcripts_b]
    if normalize_text:
        texts_a = [_strip_bracketed_artifacts(t) for t in texts_a]
        texts_b = [_strip_bracketed_artifacts(t) for t in texts_b]
        texts_a = [_normalize_numbers(t) for t in texts_a]
        texts_b = [_normalize_numbers(t) for t in texts_b]

    pairwise = [
        {"take_i": i + 1, "take_j": j + 1, "wer": compute_wer(ta, tb, normalize_text)}
        for i, ta in enumerate(texts_a)
        for j, tb in enumerate(texts_b)
    ]
    return _wer_stats(pairwise)
