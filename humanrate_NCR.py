#!/usr/bin/env python3
"""GUI for human rating of clarification requests (NCR).

Calls Gemini to propose candidate clarification requests from each participant's
transcripts, then presents a two-column view:
  • Left  — full transcript with proposed CRs highlighted in yellow
  • Right — Accept / Reject buttons for each highlighted turn

Results are saved to experiment_data_NCR_by_<alias>.json.
"""

import copy
import json
import os
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ──────────────────────────────────── constants ───────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
MODEL_NAME = "gemini-2.5-flash"  

ACCEPT = "accept"
REJECT = "reject"
UNSET = ""

CR_SYSTEM_INSTRUCTION = (
    "You are a linguistic expert identifying potential clarification requests in conversation "
    "transcripts. Mark ANY utterance from the user (usually Speaker B) (the participant/tourist/customer) that could "
    "possibly be a clarification request — err heavily on the side of inclusion.\n\n"
    "A clarification request includes any utterance where a speaker:\n"
    "- Asks to repeat or re-say something ('Can you repeat that?', 'Say that again?')\n"
    "- Asks for clarification of something already mentioned\n"
    "- Partially repeats something with a trailing or rising intonation\n"
    "- Asks for spelling, pronunciation, or confirmation of a word or phrase\n"
    "- Expresses that they didn't fully hear or understand something already said\n"
    "- Makes any request to speak louder or more clearly\n\n"
    "- Makes a sound indicating that they did not hear properly ('Huh?', 'Sorry?', 'What?')\n\n"
    "Do NOT include general information-seeking questions about new topics.\n"
    "Only mark utterances of the user (usually Speaker B) never the agent (Speaker A)."
)

CR_USER_PROMPT = (
    "Identify clarification requests in the transcript below according to the system instruction. "
    "Each line is prefixed with its index in brackets, e.g. [0], [1], etc. "
    "Return ONLY a JSON object with a single key \"cr_line_indices\" whose value is an array "
    "of the integer indices of lines that are clarification requests. "
    "Example: {\"cr_line_indices\": [2, 7, 12]}\n\n"
    "TRANSCRIPT:\n"
)

# ─────────────────────────────────── Gemini ──────────────────────────────────


def _fetch_phase_proposals(client, phase_key: str, analysis_dir: Path) -> dict:
    from google.genai import types
    from google.genai.types import ThinkingConfig

    tx_path = analysis_dir / f"transcript_{phase_key}.json"
    if not tx_path.exists():
        return {"lines": [], "cr_line_indices": []}

    segments = json.loads(tx_path.read_text(encoding="utf-8"))
    raw_lines = [f"[{s['timestamp']}] {s['speakerID']}: {s['text']}" for s in segments]
    numbered_transcript = "\n".join(f"[{i}] {line}" for i, line in enumerate(raw_lines))
    prompt = f"{CR_USER_PROMPT}{numbered_transcript}"

    config = types.GenerateContentConfig(
        system_instruction=CR_SYSTEM_INSTRUCTION,
        temperature=0.0,
        seed=42,
        candidate_count=1,
        thinking_config=ThinkingConfig(thinking_budget=0),
        response_mime_type="application/json",
    )
    response = client.models.generate_content(
        model=MODEL_NAME, contents=[prompt], config=config
    )
    result = json.loads(response.text)
    cr_indices = set(result.get("cr_line_indices", []))
    cr_lines = [
        {"index": i, "text": raw_lines[i]}
        for i in sorted(cr_indices)
        if i < len(raw_lines)
    ]
    return {"cr_lines": cr_lines}


def fetch_and_save_proposals(folder: Path) -> None:
    from google import genai

    api_key = os.getenv("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    analysis_dir = folder / "analysis"
    save_data = {}
    for phase_key, _ in [("training", "Training"), ("experiment", "Experiment")]:
        save_data[phase_key] = _fetch_phase_proposals(client, phase_key, analysis_dir)

    save_path = analysis_dir / "gemini_NCR_proposals.json"
    save_path.write_text(json.dumps(save_data, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_annotated_transcript(folder: Path, phase_key: str) -> list[dict]:
    proposals_path = folder / "analysis" / "gemini_NCR_proposals.json"
    all_data = json.loads(proposals_path.read_text(encoding="utf-8"))
    phase_data = all_data.get(phase_key, {})
    cr_indices = {item["index"] for item in phase_data.get("cr_lines", [])}

    tx_path = folder / "analysis" / f"transcript_{phase_key}.json"
    segments = json.loads(tx_path.read_text(encoding="utf-8"))
    raw_lines = [f"[{s['timestamp']}] {s['speakerID']}: {s['text']}" for s in segments]

    result = []
    for i, raw in enumerate(raw_lines):
        is_candidate = i in cr_indices
        m = re.match(r"\[(\d+:\d+)\]\s+(Speaker\s+([AB])):\s+(.*)", raw)
        if m:
            ts, label, letter, utterance = m.groups()
            result.append({"timestamp": ts, "speaker": letter, "label": label,
                           "text": utterance, "raw": raw, "is_candidate": is_candidate,
                           "proposal": raw if is_candidate else None, "var": None})
        else:
            result.append({"timestamp": None, "speaker": None, "label": None,
                           "text": raw, "raw": raw, "is_candidate": is_candidate,
                           "proposal": raw if is_candidate else None, "var": None})
    return result


def _assign_group_vars(lines: list[dict]) -> None:
    """Assign a shared tk.StringVar to each run of consecutive candidates; mark group leaders."""
    i = 0
    while i < len(lines):
        if lines[i].get("is_candidate"):
            j = i + 1
            while j < len(lines) and lines[j].get("is_candidate"):
                j += 1
            shared_var = tk.StringVar(value=UNSET)
            for k in range(i, j):
                lines[k]["var"] = shared_var
                lines[k]["group_leader"] = (k == i)
            i = j
        else:
            i += 1


# ─────────────────────────────── file helpers ─────────────────────────────────


def output_filename(alias: str) -> str:
    return f"experiment_data_NCR_by_{alias}.json"


def find_participants(alias: str) -> list[tuple[Path, dict]]:
    """Return (folder, data) for participants without an NCR output file for this alias."""
    out_name = output_filename(alias)
    result = []
    for folder in sorted(DATA_DIR.glob("experiment_*")):
        if not folder.is_dir() or (folder / "analysis" / out_name).exists():
            continue
        fpath = folder / "experiment_data.json"
        if fpath.exists():
            with open(fpath, encoding="utf-8") as fh:
                result.append((folder, json.load(fh)))
    return result


def _sanitize_alias(raw: str) -> str:
    slug = re.sub(r"\s+", "_", raw.strip())
    return re.sub(r"[^\w\-]", "", slug)


def _existing_ncr_files(alias: str) -> list[Path]:
    name = output_filename(alias)
    return [f for f in DATA_DIR.glob(f"experiment_*/analysis/{name}") if f.is_file()]


# ──────────────────────────────────── app ────────────────────────────────────


class NCRApp:
    PHASES = [("training", "Training"), ("experiment", "Experiment")]

    def __init__(self, root: tk.Tk, alias: str) -> None:
        self.root = root
        self.alias = alias
        self.participants = find_participants(alias)
        self.idx = 0
        self.all_lines: dict[str, list[dict]] = {}

        if not self.participants:
            messagebox.showinfo("Done", f"No unscored participants for alias '{alias}'.")
            root.destroy()
            return

        self._build_chrome()
        self._load_participant(0)

    # ──────────────────────────────── layout ─────────────────────────────────

    def _build_chrome(self) -> None:
        self.root.title(f"NCR Rater — {self.alias}")
        self.root.geometry("1350x820")
        self.root.minsize(900, 600)
        self.root.resizable(True, True)

        top = tk.Frame(self.root, pady=6, padx=12)
        top.pack(side=tk.TOP, fill=tk.X)

        self.status_label = tk.Label(top, text="", font=("Arial", 11, "bold"), anchor="w")
        self.status_label.pack(side=tk.LEFT)
        tk.Label(top, text=f"Expert: {self.alias}", font=("Arial", 10), fg="#555").pack(
            side=tk.LEFT, padx=16
        )
        self.loading_label = tk.Label(top, text="", font=("Arial", 10, "italic"), fg="#e07000")
        self.loading_label.pack(side=tk.LEFT, padx=8)
        self.id_label = tk.Label(top, text="", font=("Arial", 10), fg="#888", anchor="e")
        self.id_label.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=12)

        mid = tk.Frame(self.root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=4)

        self.canvas = tk.Canvas(mid, highlightthickness=0)
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = tk.Frame(self.canvas)
        self._cwin = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.inner.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self._cwin, width=e.width),
        )
        self.canvas.bind_all("<MouseWheel>", self._on_scroll)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=12)
        bot = tk.Frame(self.root, pady=10, padx=12)
        bot.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(bot, text="Quit", width=10, command=self._quit).pack(side=tk.RIGHT, padx=4)
        self.next_btn = tk.Button(
            bot, text="Next →", width=12, command=self._next, font=("Arial", 10, "bold")
        )
        self.next_btn.pack(side=tk.RIGHT, padx=4)

    def _on_scroll(self, event: tk.Event) -> None:
        if event.delta == 0:
            return
        units = max(1, abs(event.delta) // 20) * (-1 if event.delta > 0 else 1)
        self.canvas.yview_scroll(units, "units")

    # ─────────────────────────────── loading ─────────────────────────────────

    def _load_participant(self, idx: int) -> None:
        self.idx = idx
        folder, _ = self.participants[idx]
        total = len(self.participants)

        self.status_label.config(text=f"Participant {idx + 1} of {total}")
        self.id_label.config(text=f"ID: {folder.name.rsplit('_', 1)[-1]}")
        self.next_btn.config(text="Finish ✓" if idx + 1 >= total else "Next →")

        for w in self.inner.winfo_children():
            w.destroy()

        self.loading_label.config(text="⏳ Fetching Gemini analysis…")
        self.root.update()

        proposals_path = folder / "analysis" / "gemini_NCR_proposals.json"
        if not proposals_path.exists():
            try:
                fetch_and_save_proposals(folder)
            except Exception as exc:
                messagebox.showerror("Gemini Error", f"Failed to get proposals:\n{exc}")

        self.loading_label.config(text="")

        self.all_lines = {}
        for phase_key, _ in self.PHASES:
            if proposals_path.exists():
                lines = parse_annotated_transcript(folder, phase_key)
                _assign_group_vars(lines)
                self.all_lines[phase_key] = lines
            else:
                self.all_lines[phase_key] = []

        self._render()
        self.canvas.yview_moveto(0)

    # ──────────────────────────────── render ─────────────────────────────────

    def _render(self) -> None:
        f = self.inner
        row = 0

        for phase_key, phase_label in self.PHASES:
            lines = self.all_lines.get(phase_key, [])
            n_cands = sum(1 for l in lines if l["is_candidate"])

            tk.Label(
                f,
                text=f"  {phase_label}  —  {n_cands} proposed clarification request(s)",
                font=("Arial", 20, "bold"),
                bg="#c8d4e8",
                anchor="w",
            ).grid(row=row, column=0, columnspan=3, sticky="ew", padx=2, pady=(12, 0))
            row += 1

            # Sub-headers
            tk.Label(
                f, text="Transcript", font=("Arial", 9, "bold"),
                bg="#e4e4e4", anchor="w", padx=6, pady=3,
            ).grid(row=row, column=0, sticky="ew", padx=1)
            tk.Frame(f, bg="#aaaaaa", width=2).grid(row=row, column=1, sticky="ns")
            tk.Label(
                f, text="Decision", font=("Arial", 9, "bold"),
                bg="#e4e4e4", anchor="w", padx=6, pady=3,
            ).grid(row=row, column=2, sticky="ew", padx=1)
            row += 1

            if not lines:
                tk.Label(f, text="  (no transcript found)", fg="#888").grid(
                    row=row, column=0, columnspan=3, sticky="w", padx=4, pady=4
                )
                row += 1
            else:
                for line in lines:
                    self._render_line(f, row, line)
                    row += 1

        f.grid_columnconfigure(0, weight=3)
        f.grid_columnconfigure(1, minsize=2, weight=0)
        f.grid_columnconfigure(2, weight=2)

    def _render_line(self, f: tk.Frame, row: int, line: dict) -> None:
        is_cand = line.get("is_candidate", False)

        if line["speaker"] == "A":
            bg = "#d6eaff"
        elif line["speaker"] == "B":
            bg = "#fde8e8"
        else:
            bg = "#ffffff"

        if line["timestamp"] and line["label"]:
            display = f"[{line['timestamp']}] {line['label']}: {line['text']}"
        else:
            display = line["text"]

        label_kwargs = dict(
            text=display,
            wraplength=680,
            anchor="nw",
            justify="left",
            bg=bg,
            padx=8,
            pady=5,
        )
        if is_cand:
            label_kwargs["fg"] = "#000000"
            label_kwargs["font"] = ("Arial", 14, "bold")
        tk.Label(f, **label_kwargs).grid(row=row, column=0, sticky="nsew", padx=1, pady=0)

        tk.Frame(f, bg="#aaaaaa", width=2).grid(row=row, column=1, sticky="ns")

        if is_cand and line.get("group_leader", False):
            var = line["var"]
            cell = tk.Frame(f, bg=bg, padx=10, pady=4)
            cell.grid(row=row, column=2, sticky="nsew", padx=1, pady=0)
            tk.Radiobutton(
                cell,
                text="✓ Accept",
                variable=var,
                value=ACCEPT,
                bg=bg,
                activebackground=bg,
                fg="#15c515",
                font=("Arial", 12, "bold"),
                selectcolor="#c8f0c8",
            ).pack(side=tk.LEFT, padx=8)
            tk.Radiobutton(
                cell,
                text="✗ Reject",
                variable=var,
                value=REJECT,
                bg=bg,
                activebackground=bg,
                fg="#ff0000",
                font=("Arial", 12, "bold"),
                selectcolor="#f0c8c8",
            ).pack(side=tk.LEFT, padx=8)
        else:
            tk.Frame(f, bg=bg).grid(row=row, column=2, sticky="nsew", padx=1, pady=0)

    # ──────────────────────────────── actions ────────────────────────────────

    def _undecided(self) -> list[dict]:
        return [
            line
            for phase_lines in self.all_lines.values()
            for line in phase_lines
            if line.get("is_candidate") and line.get("group_leader") and line["var"].get() == UNSET
        ]

    def _next(self) -> None:
        pending = self._undecided()
        if pending:
            messagebox.showwarning(
                "Incomplete",
                f"{len(pending)} proposed clarification request(s) not yet decided.\n"
                "Please accept or reject every highlighted turn before continuing.",
            )
            return
        self._save()
        if self.idx + 1 >= len(self.participants):
            messagebox.showinfo("Done", "All participants have been rated!")
            self.root.destroy()
        else:
            self._load_participant(self.idx + 1)

    def _save(self) -> None:
        folder, data = self.participants[self.idx]

        output = copy.deepcopy(data)
        for phase_key, _ in self.PHASES:
            lines = self.all_lines.get(phase_key, [])
            accepted = [
                l["text"]
                for l in lines
                if l.get("is_candidate") and l.get("group_leader") and l["var"].get() == ACCEPT
            ]
            output[f"clarification_requests_{phase_key}"] = len(accepted)
            output[f"clarification_examples_{phase_key}"] = accepted

        out_path = folder / "analysis" / output_filename(self.alias)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)

    def _quit(self) -> None:
        choice = messagebox.askyesnocancel(
            "Quit",
            "Save what you have rated so far?\n"
            "You can resume later with the same alias.",
        )
        if choice is None:
            return
        if choice:
            if not self._undecided():
                self._save()
            else:
                messagebox.showinfo(
                    "Note",
                    "Current participant has undecided CRs and won't be saved.\n"
                    "All previously rated participants are already saved.",
                )
        self.root.destroy()


# ──────────────────────────────── startup screen ─────────────────────────────


def _ask_resume_or_scratch(parent: tk.Tk, alias: str) -> bool | None:
    result: list[bool | None] = [None]

    dlg = tk.Toplevel(parent)
    dlg.title("Resume or start over?")
    dlg.resizable(False, False)
    dlg.grab_set()

    tk.Label(
        dlg,
        text=f'Scores from expert "{alias}" already exist.\nResume or start from scratch?',
        font=("Arial", 11),
        justify="center",
        padx=30,
        pady=20,
    ).pack()

    btn_row = tk.Frame(dlg, pady=14)
    btn_row.pack()

    def choose(v: bool) -> None:
        result[0] = v
        dlg.destroy()

    tk.Button(
        btn_row, text="Resume", width=14, font=("Arial", 10, "bold"),
        command=lambda: choose(True),
    ).pack(side=tk.LEFT, padx=10)
    tk.Button(
        btn_row, text="Start from scratch", width=16,
        command=lambda: choose(False),
    ).pack(side=tk.LEFT, padx=10)

    dlg.update_idletasks()
    px, py = parent.winfo_x(), parent.winfo_y()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    dw, dh = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")
    parent.wait_window(dlg)
    return result[0]


def main() -> None:
    root = tk.Tk()
    root.title("NCR Rater")
    root.resizable(False, False)
    root.eval("tk::PlaceWindow . center")

    frame = tk.Frame(root, padx=40, pady=30)
    frame.pack(expand=True)

    tk.Label(frame, text="NCR Rater", font=("Arial", 18, "bold")).pack(pady=(0, 6))
    tk.Label(
        frame,
        text="Accept or reject Gemini-proposed clarification requests.",
        font=("Arial", 10),
        fg="#555",
    ).pack(pady=(0, 24))

    tk.Label(frame, text="Your alias (appears in the output filename):", font=("Arial", 10)).pack(
        anchor="w"
    )
    alias_var = tk.StringVar()
    entry = tk.Entry(frame, textvariable=alias_var, width=32, font=("Arial", 11))
    entry.pack(pady=(4, 16))
    entry.focus_set()

    def start() -> None:
        alias = _sanitize_alias(alias_var.get())
        if not alias:
            messagebox.showwarning(
                "Invalid alias", "Alias must contain at least one letter or digit."
            )
            return
        existing = _existing_ncr_files(alias)
        if existing:
            choice = _ask_resume_or_scratch(root, alias)
            if choice is None:
                return
            if not choice:
                for f in existing:
                    f.unlink()

        frame.destroy()
        root.resizable(True, True)
        NCRApp(root, alias)

    tk.Button(frame, text="Start →", command=start, width=14, font=("Arial", 11, "bold")).pack()
    entry.bind("<Return>", lambda *_: start())

    root.mainloop()


if __name__ == "__main__":
    main()
