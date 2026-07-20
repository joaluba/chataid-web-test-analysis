import copy
import json
import re
import string
import unicodedata
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


# Filler/backchannel spellings that should compare equal, grouped by canonical form.
# "um" = hesitation sounds; "hmm" = backchannel/acknowledgment sounds. Kept separate
# since they likely signal different things when analysing communication difficulty.
FILLER_GROUPS = {
    "um":  {"um", "umm", "ummm", "uh", "uhh", "uhm", "ehm", "erm", "er"},
    "hmm": {"hmm", "hm", "mhm", "mm", "mmm"},
}

FILLER_CANONICAL = {
    variant: canonical
    for canonical, variants in FILLER_GROUPS.items()
    for variant in variants
}


def _normalize_fillers(text):
    """Rewrite filler/backchannel word variants (e.g. "uh", "ehm") to one canonical
    spelling per group (see FILLER_GROUPS), so they compare equal regardless of
    which spelling a transcript used."""
    tokens = text.split()
    out = [
        FILLER_CANONICAL.get(tok.strip(string.punctuation).lower(), tok)
        for tok in tokens
    ]
    return " ".join(out)


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
    return sorted(Path(participant_dir).glob(f"analysis/*_{tag}*.json"))

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
        texts = [_normalize_fillers(t) for t in texts]
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
        texts_a = [_normalize_fillers(t) for t in texts_a]
        texts_b = [_normalize_fillers(t) for t in texts_b]
        texts_a = [_normalize_numbers(t) for t in texts_a]
        texts_b = [_normalize_numbers(t) for t in texts_b]

    pairwise = [
        {"take_i": i + 1, "take_j": j + 1, "wer": compute_wer(ta, tb, normalize_text)}
        for i, ta in enumerate(texts_a)
        for j, tb in enumerate(texts_b)
    ]
    return _wer_stats(pairwise)


# ── CLARIFICATION REQUEST (NCR) DATA ───────────────────────────────────────────


def _find_participant_folder(data_dir: Path, export_id: str) -> Path:
    """Match experiment_data["exportId"] back to its data/experiment_<exportId> folder.
    Compares Unicode-normalized names, since accented exportIds (e.g. "Álvaro Díaz...")
    aren't always byte-identical to the folder name on disk."""
    target = unicodedata.normalize("NFC", export_id)
    for folder in sorted(Path(data_dir).glob("experiment_*")):
        candidate = unicodedata.normalize("NFC", folder.name.removeprefix("experiment_"))
        if candidate == target:
            return folder
    raise FileNotFoundError(f"No participant folder found for exportId {export_id!r} in {data_dir}")


def add_NCR_data(experiment_data: dict, data_dir: Path = Path("data")) -> dict:
    """Return a copy of experiment_data with an added "NCR" key holding every available
    clarification-request rating for this participant, keyed by source:
      - "gemini_proposals": LLM proposals (see getNCR_gemini.py / gemini_NCR_proposals.json)
      - "gemini_decided":   LLM final judgment (gemini_NCR_decided.json)
      - "<alias>":          human rating without LLM proposals (<alias>_NCR.json)
      - "gemini_and_<alias>": human rating with LLM proposals (gemini_and_<alias>_NCR.json)
    Each value has the shape {"training": {"cr_lines": [...]}, "experiment": {"cr_lines": [...]}}.
    Sources with no matching file on disk are simply absent from "NCR"."""
    folder = _find_participant_folder(data_dir, experiment_data["exportId"])
    analysis_dir = folder / "analysis"

    ncr = {}
    for path, source in [
        (analysis_dir / "gemini_NCR_proposals.json", "gemini_proposals"),
        (analysis_dir / "gemini_NCR_decided.json", "gemini_decided"),
    ]:
        if path.exists():
            ncr[source] = json.loads(path.read_text(encoding="utf-8"))

    for path in analysis_dir.glob("*_NCR.json"):
        source = path.stem.removesuffix("_NCR")
        ncr[source] = json.loads(path.read_text(encoding="utf-8"))

    experiment_data_with_NCR = copy.deepcopy(experiment_data)
    experiment_data_with_NCR["NCR"] = ncr
    return experiment_data_with_NCR
