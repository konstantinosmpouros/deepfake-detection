"""Read-only live progress of the Optuna babysitter run. Safe to run anytime:
    python notebooks/_progress.py
Shows the babysitter's last log line, and for each pipeline the Optuna trial count +
best val AUC so far (the study DB commits after every trial) and whether metrics.json
was written this session. Touches nothing the running kernels use.
"""
import glob
import os
from datetime import datetime
from pathlib import Path

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)
HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"

log = HERE / "_babysit.log"
if log.exists():
    lines = log.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    print("babysitter:", lines[-1] if lines else "(empty)")
print("-" * 78)

ORDER = ["cnn-finetune", "vit-lora", "clip-probe", "two-stream",
         "freqcross", "srm-noise", "patch-ensemble", "dire-recon"]
for pipe in ORDER:
    db = ART / pipe / "tuning" / f"{pipe}.db"
    metrics = ART / pipe / "metrics" / "metrics.json"
    bits = []
    if db.exists():
        try:
            st = optuna.load_study(study_name=pipe, storage=f"sqlite:///{db.as_posix()}")
            done = [t for t in st.trials if t.value is not None]
            best = f"{st.best_value:.4f}" if done else "-"
            bits.append(f"trials {len(done):>2}/{len(st.trials):<2} best_auc {best}")
        except Exception as e:
            bits.append(f"(study read err: {e})")
    else:
        bits.append("no study yet")
    if metrics.exists():
        ts = datetime.fromtimestamp(metrics.stat().st_mtime).strftime("%H:%M:%S")
        bits.append(f"metrics.json @ {ts}")
    print(f"{pipe:16s} | " + " | ".join(bits))
