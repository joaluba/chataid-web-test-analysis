#!/usr/bin/env python3
"""GUI for human rating of clarification requests (NCR).

Two modes, picked on the startup screen:
  • With LLM proposals    — Gemini-proposed candidate lines (from getNCR_gemini.py's
                             gemini_NCR_proposals.json) are highlighted, to help the
                             rater spot them quickly.
  • Without LLM proposals — no highlighting, a fully independent read of the transcript.

In both modes, every user (Speaker B) turn gets its own "Is clarification request"
toggle, off by default, that the rater switches on for the turns they judge to be one.

Results are saved to:
  • gemini_and_<alias>_NCR.json (with LLM proposals)
  • <alias>_NCR.json             (without LLM proposals)
"""

import json
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

# ──────────────────────────────────── constants ───────────────────────────────

DATA_DIR = Path(__file__).parent / "data"

LINE_PATTERN = re.compile(r"\[(\d+:\d+)\]\s+(Speaker\s+([AB])):\s+(.*)")


# ─────────────────────────────── transcript parsing ───────────────────────────


def _parse_line(raw: str) -> dict:
    m = LINE_PATTERN.match(raw)
    if m:
        ts, label, letter, utterance = m.groups()
        return {"timestamp": ts, "speaker": letter, "label": label, "text": utterance}
    return {"timestamp": None, "speaker": None, "label": None, "text": raw}


def parse_transcript(folder: Path, phase_key: str, with_proposals: bool) -> list[dict]:
    tx_path = folder / "analysis" / f"transcript_{phase_key}.json"
    segments = json.loads(tx_path.read_text(encoding="utf-8"))
    raw_lines = [f"[{s['timestamp']}] {s['speakerID']}: {s['text']}" for s in segments]

    proposed_indices = set()
    if with_proposals:
        proposals_path = folder / "analysis" / "gemini_NCR_proposals.json"
        all_data = json.loads(proposals_path.read_text(encoding="utf-8"))
        phase_data = all_data.get(phase_key, {})
        proposed_indices = {item["index"] for item in phase_data.get("cr_lines", [])}

    lines = []
    for i, raw in enumerate(raw_lines):
        line = _parse_line(raw)
        line["raw"] = raw
        line["is_proposed"] = i in proposed_indices
        line["is_cr"] = tk.BooleanVar(value=False)
        lines.append(line)
    return lines


# ─────────────────────────────── file helpers ─────────────────────────────────


def ncr_only_filename(alias: str, with_proposals: bool) -> str:
    """Filename for the NCR rating file (same {phase: {cr_lines: [...]}} shape as
    gemini_NCR_proposals.json / gemini_NCR_decided.json), for direct comparison
    against the LLM outputs. Merged into experiment_data by analysis_helpers.add_NCR_data."""
    if with_proposals:
        return f"gemini_and_{alias}_NCR.json"
    return f"{alias}_NCR.json"


def find_participants(alias: str, with_proposals: bool) -> list[Path]:
    """Return folders ready to rate in this mode, that don't already have an NCR
    output file for this alias+mode."""
    out_name = ncr_only_filename(alias, with_proposals)
    result = []
    for folder in sorted(DATA_DIR.glob("experiment_*")):
        if not folder.is_dir() or (folder / "analysis" / out_name).exists():
            continue
        if with_proposals and not (folder / "analysis" / "gemini_NCR_proposals.json").exists():
            continue
        if (folder / "experiment_data.json").exists():
            result.append(folder)
    return result


def _sanitize_alias(raw: str) -> str:
    slug = re.sub(r"\s+", "_", raw.strip())
    return re.sub(r"[^\w\-]", "", slug)


def _existing_ncr_files(alias: str, with_proposals: bool) -> list[Path]:
    name = ncr_only_filename(alias, with_proposals)
    return [f for f in DATA_DIR.glob(f"experiment_*/analysis/{name}") if f.is_file()]


# ──────────────────────────────────── app ────────────────────────────────────


class NCRApp:
    PHASES = [("training", "Training"), ("experiment", "Experiment")]

    def __init__(self, root: tk.Tk, alias: str, with_proposals: bool) -> None:
        self.root = root
        self.alias = alias
        self.with_proposals = with_proposals
        self.participants = find_participants(alias, with_proposals)
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
        mode_label = "with LLM proposals" if self.with_proposals else "without LLM proposals"
        self.root.title(f"NCR Rater — {self.alias} ({mode_label})")
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
        folder = self.participants[idx]
        total = len(self.participants)

        self.status_label.config(text=f"Participant {idx + 1} of {total}")
        self.id_label.config(text=f"ID: {folder.name.rsplit('_', 1)[-1]}")
        self.next_btn.config(text="Finish ✓" if idx + 1 >= total else "Next →")

        for w in self.inner.winfo_children():
            w.destroy()

        self.all_lines = {
            phase_key: parse_transcript(folder, phase_key, self.with_proposals)
            for phase_key, _ in self.PHASES
        }

        self._render()
        self.canvas.yview_moveto(0)

    # ──────────────────────────────── render ─────────────────────────────────

    def _render(self) -> None:
        f = self.inner
        row = 0

        for phase_key, phase_label in self.PHASES:
            lines = self.all_lines.get(phase_key, [])
            header = f"  {phase_label}"
            if self.with_proposals:
                n_proposed = sum(1 for l in lines if l["is_proposed"])
                header += f"  —  {n_proposed} proposed clarification request(s)"

            tk.Label(
                f, text=header, font=("Arial", 20, "bold"), bg="#c8d4e8", anchor="w",
            ).grid(row=row, column=0, columnspan=3, sticky="ew", padx=2, pady=(12, 0))
            row += 1

            # Sub-headers
            tk.Label(
                f, text="Transcript", font=("Arial", 9, "bold"),
                bg="#e4e4e4", anchor="w", padx=6, pady=3,
            ).grid(row=row, column=0, sticky="ew", padx=1)
            tk.Frame(f, bg="#aaaaaa", width=2).grid(row=row, column=1, sticky="ns")
            tk.Label(
                f, text="Clarification request?", font=("Arial", 9, "bold"),
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
        if self.with_proposals and line["is_proposed"]:
            label_kwargs["fg"] = "#000000"
            label_kwargs["font"] = ("Arial", 14, "bold")
        tk.Label(f, **label_kwargs).grid(row=row, column=0, sticky="nsew", padx=1, pady=0)

        tk.Frame(f, bg="#aaaaaa", width=2).grid(row=row, column=1, sticky="ns")

        if line["speaker"] == "B":
            cell = tk.Frame(f, bg=bg, padx=10, pady=4)
            cell.grid(row=row, column=2, sticky="nsew", padx=1, pady=0)
            tk.Checkbutton(
                cell,
                text="Is clarification request",
                variable=line["is_cr"],
                bg=bg,
                activebackground=bg,
                selectcolor="#c8f0c8",
                font=("Arial", 11, "bold"),
            ).pack(side=tk.LEFT, padx=8)
        else:
            tk.Frame(f, bg=bg).grid(row=row, column=2, sticky="nsew", padx=1, pady=0)

    # ──────────────────────────────── actions ────────────────────────────────

    def _next(self) -> None:
        self._save()
        if self.idx + 1 >= len(self.participants):
            messagebox.showinfo("Done", "All participants have been rated!")
            self.root.destroy()
        else:
            self._load_participant(self.idx + 1)

    def _save(self) -> None:
        folder = self.participants[self.idx]

        ncr_data = {}
        for phase_key, _ in self.PHASES:
            lines = self.all_lines.get(phase_key, [])
            accepted = [
                (i, l) for i, l in enumerate(lines) if l["speaker"] == "B" and l["is_cr"].get()
            ]
            ncr_data[phase_key] = {
                "cr_lines": [{"index": i, "text": l["raw"]} for i, l in accepted]
            }

        ncr_path = folder / "analysis" / ncr_only_filename(self.alias, self.with_proposals)
        with open(ncr_path, "w", encoding="utf-8") as fh:
            json.dump(ncr_data, fh, indent=2, ensure_ascii=False)

    def _quit(self) -> None:
        choice = messagebox.askyesnocancel(
            "Quit",
            "Save what you have rated so far?\n"
            "You can resume later with the same alias and mode.",
        )
        if choice is None:
            return
        if choice:
            self._save()
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
        text="Mark which turns are clarification requests.",
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

    tk.Label(frame, text="Mode:", font=("Arial", 10)).pack(anchor="w")
    mode_var = tk.StringVar(value="with_proposals")
    tk.Radiobutton(
        frame, text="With LLM proposals (highlighted)", variable=mode_var,
        value="with_proposals", font=("Arial", 10),
    ).pack(anchor="w", padx=8)
    tk.Radiobutton(
        frame, text="Without LLM proposals (independent)", variable=mode_var,
        value="without_proposals", font=("Arial", 10),
    ).pack(anchor="w", padx=8, pady=(0, 16))

    def start() -> None:
        alias = _sanitize_alias(alias_var.get())
        if not alias:
            messagebox.showwarning(
                "Invalid alias", "Alias must contain at least one letter or digit."
            )
            return
        with_proposals = mode_var.get() == "with_proposals"

        existing = _existing_ncr_files(alias, with_proposals)
        if existing:
            choice = _ask_resume_or_scratch(root, alias)
            if choice is None:
                return
            if not choice:
                for f in existing:
                    f.unlink()

        frame.destroy()
        root.resizable(True, True)
        NCRApp(root, alias, with_proposals)

    tk.Button(frame, text="Start →", command=start, width=14, font=("Arial", 11, "bold")).pack()
    entry.bind("<Return>", lambda *_: start())

    root.mainloop()


if __name__ == "__main__":
    main()
