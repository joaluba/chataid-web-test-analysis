#!/usr/bin/env python3
"""GUI for rating participant answers (ICS: Information Collection Score).

Reads experiment_data_plus_analysis.json (or experiment_data.json) for each
participant, shows a rating table, and saves scores to
experiment_data_ICS_by_<alias>.json.
"""

import copy
import json
import re
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

DATA_DIR = Path(__file__).parent / "data"

UNSET = -1.0

GROUND_TRUTH: dict = {
    "training": {
        "Price of a single metro ticket": "2.5 euros",
        "Current sea water temperature": "15 degrees",
        "Which museums have free admission today": "Museum of Pablo Picasso, Museum of Design, Museum of Science",
        "Tourist office closing time": "10pm",
    },
    "experiment": {
        "Price of a coffee with milk": "1.9 euros",
        "Milk options available": "Almond, Coconut, Cow Milk",
        "Is vegan milk more expensive?": "Yes, 20 cents extra",
        "What is the cafe's specialty cake?": "Tarta de Santiago (almond cake)",
        "Wi-Fi network name": "Coffee And Jazz (written together)",
        "Wi-Fi password": "Enjoy Your Coffee (written together)",
        "Maximum table usage duration": "90 minutes",
        "Evening event": "Jazz concert",
        "Artist's name": "Barcelona Jazz Collective",
        "Cafe closing time": "2am",
    },
}


def output_filename(alias: str) -> str:
    return f"experiment_data_ICS_by_{alias}.json"


def find_participants(alias: str) -> list[tuple[Path, dict]]:
    """Return (folder, data) for every participant lacking an ICS file for this alias."""
    out_name = output_filename(alias)
    result = []
    for folder in sorted(DATA_DIR.glob("experiment_*")):
        if not folder.is_dir():
            continue
        if (folder / out_name).exists():
            continue
        fpath = folder / "experiment_data.json"
        if fpath.exists():
            with open(fpath, encoding="utf-8") as fh:
                result.append((folder, json.load(fh)))
    return result


class RatingApp:
    def __init__(self, root: tk.Tk, alias: str) -> None:
        self.root = root
        self.alias = alias
        self.participants = find_participants(alias)
        self.idx = 0
        self.training_vars: dict[str, tk.DoubleVar] = {}
        self.experiment_vars: dict[str, tk.DoubleVar] = {}

        if not self.participants:
            messagebox.showinfo("Done", f"No unscored participants found for alias '{alias}'.")
            root.destroy()
            return

        self._build_chrome()
        self._show(0)

    # ------------------------------------------------------------------ layout

    def _build_chrome(self) -> None:
        self.root.title(f"ICS Rater — {self.alias}")
        self.root.geometry("1050x740")
        self.root.minsize(800, 500)
        self.root.resizable(True, True)

        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, pady=6, padx=12)
        top.pack(side=tk.TOP, fill=tk.X)

        self.status_label = tk.Label(top, text="", font=("Arial", 11, "bold"), anchor="w")
        self.status_label.pack(side=tk.LEFT)

        tk.Label(top, text=f"Expert: {self.alias}", font=("Arial", 10), fg="#555").pack(side=tk.LEFT, padx=16)

        self.id_label = tk.Label(top, text="", font=("Arial", 10), fg="#888", anchor="e")
        self.id_label.pack(side=tk.RIGHT)

        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=12)

        # ── scrollable content ───────────────────────────────────────────────
        mid = tk.Frame(self.root)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=4)

        self.canvas = tk.Canvas(mid, highlightthickness=0)
        vsb = ttk.Scrollbar(mid, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.inner = tk.Frame(self.canvas)
        self._cwin = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(
            self._cwin, width=e.width))
        self.canvas.bind_all("<MouseWheel>", self._on_scroll)

        # ── bottom buttons ───────────────────────────────────────────────────
        ttk.Separator(self.root, orient="horizontal").pack(fill=tk.X, padx=12)
        bot = tk.Frame(self.root, pady=10, padx=12)
        bot.pack(side=tk.BOTTOM, fill=tk.X)

        tk.Button(bot, text="Quit", width=10, command=self._quit).pack(side=tk.RIGHT, padx=4)
        self.next_btn = tk.Button(bot, text="Next →", width=12, command=self._next,
                                  font=("Arial", 10, "bold"))
        self.next_btn.pack(side=tk.RIGHT, padx=4)

    def _on_scroll(self, event: tk.Event) -> None:
        delta = event.delta
        if delta == 0:
            return
        units = max(1, abs(delta) // 20) * (-1 if delta > 0 else 1)
        self.canvas.yview_scroll(units, "units")

    # --------------------------------------------------------------- rendering

    def _show(self, idx: int) -> None:
        self.idx = idx
        folder, data = self.participants[idx]

        for w in self.inner.winfo_children():
            w.destroy()
        self.training_vars.clear()
        self.experiment_vars.clear()

        total = len(self.participants)
        self.status_label.config(text=f"Participant {idx + 1} of {total}")
        participant_id = folder.name.rsplit("_", 1)[-1]
        self.id_label.config(text=f"ID: {participant_id}")
        self.next_btn.config(text="Finish ✓" if idx + 1 >= total else "Next →")

        self._add_section(
            "Training",
            data.get("training_userinput", {}),
            GROUND_TRUTH["training"],
            self.training_vars,
            row_offset=0,
        )
        n_training = len(data.get("training_userinput", {}))
        self._add_section(
            "Experiment",
            data.get("experiment_userinput", {}),
            GROUND_TRUTH["experiment"],
            self.experiment_vars,
            row_offset=n_training + 3,
        )

        self.canvas.yview_moveto(0)

    def _add_section(
        self,
        title: str,
        userinput: dict,
        gt_section: dict,
        vars_dict: dict,
        row_offset: int,
    ) -> None:
        f = self.inner

        tk.Label(
            f, text=f"  {title}",
            font=("Arial", 10, "bold"), bg="#c8d4e8", anchor="w",
        ).grid(row=row_offset, column=0, columnspan=4, sticky="ew", padx=2, pady=(10, 0))

        headers = [("Question", 28), ("Ground Truth", 22), ("Participant Answer", 24), ("Score", 14)]
        for col, (text, width) in enumerate(headers):
            tk.Label(
                f, text=text, font=("Arial", 9, "bold"),
                bg="#e4e4e4", width=width, anchor="w", padx=6, pady=3,
            ).grid(row=row_offset + 1, column=col, sticky="ew", padx=1, pady=0)

        for i, (question, answer) in enumerate(userinput.items()):
            r = row_offset + 2 + i
            bg = "#ffffff" if i % 2 == 0 else "#f7f7f7"
            gt_text = gt_section.get(question, "")

            tk.Label(f, text=question, wraplength=220, width=28, anchor="nw",
                     justify="left", bg=bg, padx=6, pady=5).grid(
                row=r, column=0, sticky="nsew", padx=1, pady=0)
            tk.Label(f, text=gt_text, wraplength=175, width=22, anchor="nw",
                     justify="left", bg=bg, fg="#1a5c1a", padx=6, pady=5).grid(
                row=r, column=1, sticky="nsew", padx=1, pady=0)
            tk.Label(f, text=answer, wraplength=190, width=24, anchor="nw",
                     justify="left", bg=bg, padx=6, pady=5).grid(
                row=r, column=2, sticky="nsew", padx=1, pady=0)

            var = tk.DoubleVar(value=UNSET)
            vars_dict[question] = var

            score_cell = tk.Frame(f, bg=bg, padx=6, pady=3)
            score_cell.grid(row=r, column=3, sticky="nsew", padx=1, pady=0)
            for val, label in ((0.0, "0"), (0.25, "0.25"), (0.5, "0.5"), (0.75, "0.75"), (1.0, "1")):
                tk.Radiobutton(
                    score_cell, text=label, variable=var, value=val,
                    bg=bg, activebackground=bg,
                ).pack(side=tk.LEFT, padx=2)

        for col in range(4):
            f.grid_columnconfigure(col, weight=1)

    # ----------------------------------------------------------------- actions

    def _next(self) -> None:
        unset = [q for q, v in {**self.training_vars, **self.experiment_vars}.items()
                 if v.get() == UNSET]
        if unset:
            msg = f"{len(unset)} answer(s) not yet scored:\n" + "\n".join(f"  • {q}" for q in unset)
            messagebox.showwarning("Incomplete", msg)
            return

        self._save()

        if self.idx + 1 >= len(self.participants):
            messagebox.showinfo("Done", "All participants have been scored!")
            self.root.destroy()
        else:
            self._show(self.idx + 1)

    def _save(self) -> None:
        folder, data = self.participants[self.idx]

        training_scores = {q: v.get() for q, v in self.training_vars.items()}
        experiment_scores = {q: v.get() for q, v in self.experiment_vars.items()}

        output = copy.deepcopy(data)
        analysis = output.setdefault("analysis", {})
        analysis["training_userinput_scores"] = training_scores
        analysis["experiment_userinput_scores"] = experiment_scores
        analysis["training_score_total"] = sum(training_scores.values())
        analysis["experiment_score_total"] = sum(experiment_scores.values())

        out_path = folder / output_filename(self.alias)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(output, fh, indent=2, ensure_ascii=False)

    def _quit(self) -> None:
        save = messagebox.askyesnocancel(
            "Quit",
            "Do you want to save what you have rated so far?\n"
            "You will be able to resume rating with the same alias later on.",
        )
        if save is None:
            return  # Cancel — stay in the app
        if save:
            all_scored = all(v.get() != UNSET
                             for v in {**self.training_vars, **self.experiment_vars}.values())
            if all_scored:
                self._save()
            else:
                messagebox.showinfo(
                    "Note",
                    "The current participant is not fully scored and won't be saved.\n"
                    "All previously rated participants are already saved.",
                )
        self.root.destroy()


# ──────────────────────────────────────────────────────────────── startup screen


def _sanitize_alias(raw: str) -> str:
    """Keep alphanumerics and hyphens; collapse spaces to underscores."""
    slug = re.sub(r"\s+", "_", raw.strip())
    slug = re.sub(r"[^\w\-]", "", slug)
    return slug


def _existing_ics_files(alias: str) -> list[Path]:
    """Return all ICS output files that already exist for this alias."""
    name = output_filename(alias)
    return [f for f in DATA_DIR.glob(f"experiment_*/{name}") if f.is_file()]


def _ask_resume_or_scratch(parent: tk.Tk, alias: str) -> bool | None:
    """Show a modal dialog asking Resume vs Start from scratch.

    Returns True  = resume
            False = start from scratch
            None  = cancelled (window closed)
    """
    result: list[bool | None] = [None]

    dlg = tk.Toplevel(parent)
    dlg.title("Resume or start over?")
    dlg.resizable(False, False)
    dlg.grab_set()  # modal

    tk.Label(
        dlg,
        text=f"There are already scores from the expert  \"{alias}\".\nDo you want to resume or start from scratch?",
        font=("Arial", 11),
        justify="center",
        padx=30,
        pady=20,
    ).pack()

    btn_row = tk.Frame(dlg, pady=14)
    btn_row.pack()

    def choose(value: bool) -> None:
        result[0] = value
        dlg.destroy()

    tk.Button(btn_row, text="Resume", width=14, font=("Arial", 10, "bold"),
              command=lambda: choose(True)).pack(side=tk.LEFT, padx=10)
    tk.Button(btn_row, text="Start from scratch", width=16, font=("Arial", 10),
              command=lambda: choose(False)).pack(side=tk.LEFT, padx=10)

    # Centre over parent
    dlg.update_idletasks()
    px, py = parent.winfo_x(), parent.winfo_y()
    pw, ph = parent.winfo_width(), parent.winfo_height()
    dw, dh = dlg.winfo_width(), dlg.winfo_height()
    dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")

    parent.wait_window(dlg)
    return result[0]


def main() -> None:
    root = tk.Tk()
    root.title("ICS Rater")
    root.resizable(False, False)
    root.eval("tk::PlaceWindow . center")

    # ── alias entry frame ────────────────────────────────────────────────────
    frame = tk.Frame(root, padx=40, pady=30)
    frame.pack(expand=True)

    tk.Label(frame, text="ICS Rater", font=("Arial", 18, "bold")).pack(pady=(0, 6))
    tk.Label(frame, text="Rate participant answers for information collection scores.",
             font=("Arial", 10), fg="#555").pack(pady=(0, 24))

    tk.Label(frame, text="Your alias (will appear in the output filename):",
             font=("Arial", 10)).pack(anchor="w")

    alias_var = tk.StringVar()
    entry = tk.Entry(frame, textvariable=alias_var, width=32, font=("Arial", 11))
    entry.pack(pady=(4, 16))
    entry.focus_set()

    def start() -> None:
        raw = alias_var.get()
        alias = _sanitize_alias(raw)
        if not alias:
            messagebox.showwarning("Invalid alias",
                                   "Alias must contain at least one letter or digit.")
            return

        existing = _existing_ics_files(alias)
        if existing:
            choice = _ask_resume_or_scratch(root, alias)
            if choice is None:
                return  # dialog closed without choosing
            if not choice:  # start from scratch — delete previous scores
                for f in existing:
                    f.unlink()

        frame.destroy()
        root.resizable(True, True)
        RatingApp(root, alias)

    tk.Button(frame, text="Start →", command=start, width=14,
              font=("Arial", 11, "bold")).pack()
    entry.bind("<Return>", lambda *_: start())

    root.mainloop()


if __name__ == "__main__":
    main()
