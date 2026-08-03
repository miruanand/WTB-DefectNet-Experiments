r"""
run_experiment.py
==================
Run this ONE script and leave the computer on. It will:

    1. Train WTB-DefectNet into ./runs/<exp_name>/            (train.py)
    2. Evaluate best.pt on the test set                        (evaluate.py)
    3. Plot epoch-wise loss/accuracy/F1/balanced-acc curves    (plot_curves.py)
    4. Generate Grad-CAM XAI figures for the final model       (gradcam.py)
    5. git add / commit / push the whole ./runs/<exp_name>/ folder
       to your GitHub repo, so it shows up as a new folder
       (e.g. "Try_2") without you touching the keyboard again.

USAGE (from the repo root, same folder as train.py):

    python run_experiment.py --data_root "D:\...\WTBs2025" --exp_name Try_2

If training gets interrupted (sleep, crash, power loss) just re-run the
EXACT SAME command -- train.py's own --resume support means it picks up
from the last fully completed epoch, not from scratch.

Every subprocess's full console output is also written to
./runs/<exp_name>/pipeline_log.txt, so if you come back in the morning and
something failed partway, you can see exactly what happened without having
to reproduce it.

Requirements this script assumes are already true (check BEFORE leaving it
running overnight):
    - `git` is on PATH and this folder is already a git repo with a remote
      ("origin") pointing at your GitHub repo.
    - You can already push to that remote WITHOUT typing a password/PAT
      interactively (SSH key configured, or a Windows Git Credential
      Manager entry already saved from a previous manual push/login).
      If push requires typing something, this script will hang waiting
      for input that will never come.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--exp_name", type=str, default="Try_2",
                     help="New folder name under ./runs/, also used as the git commit message tag.")
    ap.add_argument("--repo_dir", type=str, default=".",
                     help="Root of the git repo (where .git lives). Default: current directory.")
    ap.add_argument("--branch", type=str, default=None,
                     help="Git branch to push to. Default: whatever branch is currently checked out.")
    ap.add_argument("--epochs", type=int, default=None,
                     help="Override Config.epochs for this run, e.g. --epochs 70.")
    ap.add_argument("--num_per_class_gradcam", type=int, default=2)
    ap.add_argument("--skip_git", action="store_true",
                     help="Run the pipeline but don't touch git at all (useful for a dry run).")
    ap.add_argument("--push_retries", type=int, default=5,
                     help="How many times to retry `git push` if it fails (e.g. brief network drop).")
    ap.add_argument("--train_retries", type=int, default=5,
                     help="How many times to re-launch train.py --resume if it crashes "
                          "mid-run (e.g. a Windows access violation). Each retry picks up "
                          "from the last completed epoch, it does not restart from scratch.")
    ap.add_argument("--require_gpu", action="store_true",
                     help="Pass through to train.py: hard-stop if no CUDA GPU is found, "
                          "instead of quietly training on CPU for hours.")
    ap.add_argument("--backbone", type=str, default=None,
                     choices=["dsps", "resnet18", "resnet34"],
                     help="Pass through to train.py. 'dsps' (default) = fully custom "
                          "WTBDefectNet from scratch. 'resnet18'/'resnet34' = ImageNet-"
                          "pretrained stem + custom TSDB/ASA/DRFB/WGFR/MSCA/LTCP on top.")
    ap.add_argument("--unfreeze_stem", action="store_true",
                     help="Pass through to train.py (only relevant with --backbone resnet18/34).")
    ap.add_argument("--no_pretrained_stem", action="store_true",
                     help="Pass through to train.py (only relevant with --backbone resnet18/34).")
    ap.add_argument("--use_weighted_sampler", action="store_true",
                     help="Pass through to train.py: restore the old always-on sampler.")
    ap.add_argument("--num_workers", type=int, default=None,
                     help="Pass through to train.py. On Windows, DataLoader workers "
                          "sharing memory-mapped tensors can crash with 'Couldn't open "
                          "shared file mapping' or run out of shared memory after a few "
                          "epochs -- try --num_workers 0 or 2 if you hit that.")
    return ap.parse_args()


def run_step(cmd, log_file, cwd):
    """Run one subprocess, streaming its output to console AND log_file.
    Returns True on success (exit code 0), False otherwise -- never raises,
    so one failed step doesn't kill the whole overnight run."""
    banner = f"\n{'='*70}\n[{datetime.now().isoformat(timespec='seconds')}] RUNNING: {' '.join(cmd)}\n{'='*70}\n"
    print(banner)
    log_file.write(banner)
    log_file.flush()

    env = os.environ.copy()
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # belt-and-suspenders vs. the OMP #15 crash
    env["PYTHONUNBUFFERED"] = "1"          # extra guarantee alongside python -u below

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        for line in proc.stdout:
            print(line, end="")
            log_file.write(line)
        proc.wait()
        log_file.flush()
        if proc.returncode != 0:
            msg = f"\n[run_experiment] STEP FAILED (exit code {proc.returncode}): {' '.join(cmd)}\n"
            print(msg)
            log_file.write(msg)
            return False
        return True
    except FileNotFoundError as e:
        msg = f"\n[run_experiment] STEP FAILED TO START: {e}\n"
        print(msg)
        log_file.write(msg)
        return False


def git(args, cwd, log_file):
    return run_step(["git"] + args, log_file, cwd)


def main():
    args = parse_args()
    py = sys.executable  # use the same python interpreter that's running this script
    py_u = [py, "-u"]    # -u: unbuffered stdout, so pipeline_log.txt captures live
                          # progress instead of losing it if a child process crashes hard
                          # (this is why last run's log.csv showed 41 completed epochs but
                          # pipeline_log.txt showed almost nothing -- buffered output that
                          # never got flushed before the access-violation crash killed it)
    repo_dir = os.path.abspath(args.repo_dir)
    run_dir = os.path.join(repo_dir, "runs", args.exp_name)
    os.makedirs(run_dir, exist_ok=True)

    log_path = os.path.join(run_dir, "pipeline_log.txt")
    with open(log_path, "a") as log_file:

        overall_start = time.time()
        log_file.write(f"\n\n########## PIPELINE START {datetime.now().isoformat()} "
                        f"(exp_name={args.exp_name}) ##########\n")

        # ---------------- 1. TRAIN (with auto-retry on crash) ----------------
        train_cmd = py_u + [
            "train.py",
            "--data_root", args.data_root,
            "--exp_name", args.exp_name,
            "--resume",   # safe even on a totally fresh run -- no-op if no checkpoint exists yet
        ]
        if args.epochs is not None:
            train_cmd += ["--epochs", str(args.epochs)]
        if args.require_gpu:
            train_cmd += ["--require_gpu"]
        if args.backbone is not None:
            train_cmd += ["--backbone", args.backbone]
        if args.unfreeze_stem:
            train_cmd += ["--unfreeze_stem"]
        if args.no_pretrained_stem:
            train_cmd += ["--no_pretrained_stem"]
        if args.use_weighted_sampler:
            train_cmd += ["--use_weighted_sampler"]
        if args.num_workers is not None:
            train_cmd += ["--num_workers", str(args.num_workers)]

        train_ok = False
        for attempt in range(1, args.train_retries + 1):
            print(f"\n[run_experiment] Training attempt {attempt}/{args.train_retries}...")
            if run_step(train_cmd, log_file, repo_dir):
                train_ok = True
                break
            print(f"[run_experiment] train.py exited non-zero (crash, or a genuine "
                  f"early stop/error). Re-launching with --resume in 15s -- it will "
                  f"pick up from the last completed epoch, not restart from scratch.")
            time.sleep(15)

        if not train_ok:
            print(f"[run_experiment] Training still failing after {args.train_retries} "
                  f"attempts. Will still try to commit/push whatever checkpoints/logs "
                  f"exist so far, then stop.")

        best_ckpt = os.path.join(run_dir, "checkpoints", "best.pt")
        if os.path.isfile(best_ckpt):
            # ---------------- 2. EVALUATE (test set, ONE time) ----------------
            eval_cmd = py_u + [
                "evaluate.py",
                "--data_root", args.data_root,
                "--checkpoint", best_ckpt,
                "--out_dir", run_dir,
            ]
            run_step(eval_cmd, log_file, repo_dir)   # continue even if this fails -- still commit what we have

            # ---------------- 3. CURVES ----------------
            curves_cmd = py_u + ["plot_curves.py", "--run_dir", run_dir]
            run_step(curves_cmd, log_file, repo_dir)

            # ---------------- 4. GRAD-CAM (XAI, final model) ----------------
            gradcam_cmd = py_u + [
                "gradcam.py",
                "--data_root", args.data_root,
                "--checkpoint", best_ckpt,
                "--out_dir", run_dir,
                "--num_per_class", str(args.num_per_class_gradcam),
            ]
            run_step(gradcam_cmd, log_file, repo_dir)
        else:
            print(f"[run_experiment] No checkpoint at {best_ckpt} yet -- skipping "
                  f"evaluate/curves/gradcam this time, but still pushing log.csv/"
                  f"pipeline_log.txt so you can see progress remotely.")

        elapsed = (time.time() - overall_start) / 60
        print(f"\n[run_experiment] Pipeline steps finished in {elapsed:.1f} min. "
              f"Results are in: {run_dir}")

        # ---------------- 5. GIT ADD / COMMIT / PUSH ----------------
        if args.skip_git:
            print("[run_experiment] --skip_git set, not touching git.")
            return

        rel_run_dir = os.path.relpath(run_dir, repo_dir)
        git(["add", rel_run_dir], repo_dir, log_file)

        commit_msg = (
            f"Add {args.exp_name} results (auto-committed by run_experiment.py)"
            if train_ok else
            f"WIP {args.exp_name}: partial results, training did not finish "
            f"(auto-committed by run_experiment.py)"
        )
        committed = git(["commit", "-m", commit_msg], repo_dir, log_file)
        if not committed:
            print("[run_experiment] Nothing to commit (or commit failed) -- "
                  "check pipeline_log.txt. Skipping push.")
            return

        push_args = ["push", "origin"]
        if args.branch:
            push_args.append(args.branch)

        pushed = False
        for attempt in range(1, args.push_retries + 1):
            print(f"[run_experiment] git push attempt {attempt}/{args.push_retries}...")
            if git(push_args, repo_dir, log_file):
                pushed = True
                break
            time.sleep(30)  # brief backoff, e.g. for a momentary network drop

        if pushed and train_ok:
            print(f"[run_experiment] DONE. '{args.exp_name}' is committed and pushed to GitHub.")
        elif pushed and not train_ok:
            print(f"[run_experiment] PARTIAL run pushed to GitHub (training did not "
                  f"finish -- re-run the same command later to continue from epoch "
                  f"{args.exp_name} last left off at; it will resume, not restart).")
        else:
            print(f"[run_experiment] Commit succeeded locally but push failed after "
                  f"{args.push_retries} attempts. Results are safe in your local git history "
                  f"-- run `git push` manually next time you're at the machine.")


if __name__ == "__main__":
    main()
