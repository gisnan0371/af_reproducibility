from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("0",   "r2_station_topology_check.py", "station topology"),
    ("0.5", "r2_flow_audit_fix.py",         "flow audit"),
    ("0.6", "r2_match_yi_s4.py",            "Yi S4 day filters"),
    ("1",   "r2_make_model_frame.py",       "model frame"),
    ("1.5", "r2_measured_coverage.py",      "coverage checks"),
    ("2",   "r2_events_audit.py",           "events"),
    ("2.5", "r2_wq_resolution_check.py",    "NH3-N value check"),
    ("2.6", "r2_nh3_provenance.py",         "NH3-N raw record"),
    ("2.7", "af_alignment_sensitivity.py",  "AF date-alignment sensitivity"),
    ("3",   "r2_gamm_sensitivity.R",        "models (R)"),
    ("6",   "generate_all_figures_v5_9.py", "figures and tables"),
]
OPT_IN = {"0": "topology"}


def find_rscript():
    p = shutil.which("Rscript")
    if p:
        return p
    cp = os.environ.get("CONDA_PREFIX")
    cands = []
    if cp:
        cands += [os.path.join(cp, "Scripts", "Rscript.exe"), os.path.join(cp, "Lib", "R", "bin", "Rscript.exe"),
                  os.path.join(cp, "bin", "Rscript")]
    for root in (r"C:\Program Files\R", os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "R")):
        if os.path.isdir(root):
            for v in sorted(os.listdir(root), reverse=True):
                cands.append(os.path.join(root, v, "bin", "Rscript.exe"))
    return next((c for c in cands if os.path.isfile(c)), None)


def run(cmd, log, title):
    print(f"\n== {title} ({log})", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        p = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                             encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        for line in p.stdout:
            sys.stdout.write(line); fh.write(line)
        p.wait()
    return p.returncode == 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topology", action="store_true", help="also run step 0")
    ap.add_argument("--boot", type=int, default=0, help="bootstrap replicates for step 3")
    ap.add_argument("--only", nargs="*", default=None, help="step ids to run, e.g. --only 1 3 6")
    a = ap.parse_args()

    os.chdir(HERE)
    if not os.path.isfile(os.path.join(HERE, "master_recompute.py")):
        print("master_recompute.py not found in this folder")
        return 1
    try:
        subprocess.run([sys.executable, "-c", "import master_recompute"], cwd=HERE, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError:
        print("import master_recompute failed")
        return 1
    rscript = find_rscript()
    print(f"Rscript: {rscript or 'not found'}")

    if a.only:
        wanted = set(a.only)
    else:
        wanted = {sid for sid, _, _ in STEPS
                  if sid not in OPT_IN or getattr(a, OPT_IN[sid])}
    ok, skipped = [], []
    for sid, script, title in STEPS:
        if sid not in wanted:
            continue
        if not os.path.isfile(os.path.join(HERE, script)):
            print(f"skip {sid}: {script} not found"); skipped.append(sid); continue
        log = f"r2_log_step{sid.replace('.', '_')}.txt"
        if script.endswith(".R"):
            if not rscript:
                print(f"skip {sid}: R not found"); skipped.append(sid); continue
            os.environ["N_BOOT"] = str(a.boot)
            cmd = [rscript, script]
        else:
            cmd = [sys.executable, script]
        if run(cmd, log, f"{sid} {title}"):
            ok.append(sid)
        else:
            print(f"step {sid} failed, see {log}"); return 1

    print("\ndone:", ", ".join(ok) or "-")
    if skipped:
        print("skipped:", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    sys.exit(main())
