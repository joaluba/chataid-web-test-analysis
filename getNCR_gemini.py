import json
from pathlib import Path

from my_gemini_API_calls import propose_clarification_requests, judge_clarification_requests

DATA_DIR = Path("data")
PHASES = [("training", "Training"), ("experiment", "Experiment")]


def _numbered_transcript(analysis_dir: Path, phase_key: str) -> tuple[list[str], str] | None:
    tx_path = analysis_dir / f"transcript_{phase_key}.json"
    if not tx_path.exists():
        return None

    segments = json.loads(tx_path.read_text(encoding="utf-8"))
    raw_lines = [f"[{s['timestamp']}] {s['speakerID']}: {s['text']}" for s in segments]
    numbered_transcript = "\n".join(f"[{i}] {line}" for i, line in enumerate(raw_lines))
    return raw_lines, numbered_transcript


def _cr_data(raw_lines: list[str], cr_indices: set[int]) -> dict:
    cr_lines = [
        {"index": i, "text": raw_lines[i]}
        for i in sorted(cr_indices)
        if i < len(raw_lines)
    ]
    return {"cr_lines": cr_lines}


def _save_ncr(folder: Path, output_name: str, get_indices_fn) -> None:
    output_path = folder / "analysis" / output_name
    # if output_path.exists():
    #     print(f"  ℹ️  Skipping {output_path} (already exists)")
    #     return

    analysis_dir = folder / "analysis"
    save_data = {}
    for phase_key, _ in PHASES:
        result = _numbered_transcript(analysis_dir, phase_key)
        if result is None:
            save_data[phase_key] = {"cr_lines": []}
            continue
        raw_lines, numbered_transcript = result
        cr_indices = get_indices_fn(numbered_transcript)
        save_data[phase_key] = _cr_data(raw_lines, cr_indices)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  ✓ Saved to {output_path}")


for folder in sorted(DATA_DIR.iterdir()):
    if not folder.is_dir():
        continue

    print(f"\n{'='*60}")
    print(f"Processing folder: {folder.name}")
    print(f"{'='*60}")

    _save_ncr(folder, "gemini_NCR_proposals.json", propose_clarification_requests)
    _save_ncr(folder, "gemini_NCR_decided.json", judge_clarification_requests)

print(f"\n{'='*60}")
print("All folders processed!")
print(f"{'='*60}")
