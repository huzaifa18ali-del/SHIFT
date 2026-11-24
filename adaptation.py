"""Task 4: adaptation after a force-law change (k_old=1.0 -> k_new=9.0).

Design (see README): old-world pretraining; new-world data from separate seeds; budgets are nested
prefixes of one 128-trajectory new-world pool; every learned MLP keeps the ORIGINAL (old-world,
train-fitted) normalisers so fine-tuned and fresh MLPs are comparable. Validation = new-world val
set only (checkpoint selection); new-world test is used for final evaluation only.
Usage: python adaptation.py [--seeds 0 1 2] [--epochs 100] [--smoke] [--out outputs/adaptation]
"""
import argparse, json, os, platform
import numpy as np
import torch, sklearn
from sklearn.linear_model import LinearRegression

from simulator import restoring_force, generate_dataset, export_transitions
from train import TrainConfig, Normalizer, train_mlp, predict_mlp, params_fingerprint
from evaluate import evaluate, relative_improvement

K_OLD, K_NEW = 1.0, 9.0
CENTER = (0.0, 0.0)
DT, N_STEPS = 0.01, 200
BUDGETS = (8, 16, 32, 64, 128)
SIZES = {"train": 128, "val": 32, "test": 32}
METRICS = ("accel_mse", "accel_mae", "next_state_mse")


# ---------- data ----------
def seed_plan(s):
    """Six distinct data seeds per experiment seed s (s=0 -> old world 0/1/2 as in the baseline)."""
    b = 100 * s
    return {"old": {"train": b, "val": b + 1, "test": b + 2},
            "new": {"train": b + 10, "val": b + 11, "test": b + 12}}


def force_for(k):
    return restoring_force(k, *CENTER)


def make_world(k, seeds, sizes=SIZES, n_steps=N_STEPS, dt=DT):
    """{split: {traj (T+1,N,4), cur, nxt, acc}}; each split is whole trajectories from its own seed."""
    f, out = force_for(k), {}
    for split, seed in seeds.items():
        traj, (cur, nxt, acc) = generate_dataset(f, sizes[split], n_steps, dt, seed)
        out[split] = {"traj": traj, "cur": cur, "nxt": nxt, "acc": acc}
    return out


def share_initial_states(traj_a, traj_b):
    a = {tuple(r) for r in traj_a[0]}
    return any(tuple(r) in a for r in traj_b[0])


def assert_disjoint(old, new):
    named = [(f"old_{k}", v["traj"]) for k, v in old.items()] + [(f"new_{k}", v["traj"]) for k, v in new.items()]
    for i in range(len(named)):
        for j in range(i + 1, len(named)):
            if share_initial_states(named[i][1], named[j][1]):
                raise AssertionError(f"trajectories overlap: {named[i][0]} vs {named[j][0]}")


def budget_transitions(train_split, b, k=K_NEW):
    """First b whole trajectories of the new-world pool -> transitions (nested budgets)."""
    if b > train_split["traj"].shape[1]:
        raise ValueError("budget exceeds available trajectories")
    traj = train_split["traj"][:, :b]
    cur, nxt, acc = export_transitions(traj, force_for(k))
    return {"traj": traj, "cur": cur, "nxt": nxt, "acc": acc}


# ---------- evaluation ----------
def score(split, acc_pred, dt=DT):
    return evaluate(split["cur"], split["nxt"], split["acc"], acc_pred, dt)


def mlp_scores(model, split, xn, yn):
    return score(split, predict_mlp(model, split["cur"], xn, yn))


def improvement(unchanged, current):
    return {m: relative_improvement(unchanged[m], current[m]) for m in METRICS}


# ---------- one experiment seed ----------
def run_seed(seed, budgets=BUDGETS, epochs=100, pretrain_epochs=None, hidden=64, lr=1e-3,
             batch_size=256, sizes=SIZES, n_steps=N_STEPS):
    plan = seed_plan(seed)
    old = make_world(K_OLD, plan["old"], sizes, n_steps)
    new = make_world(K_NEW, plan["new"], sizes, n_steps)
    assert_disjoint(old, new)

    xn, yn = Normalizer().fit(old["train"]["cur"]), Normalizer().fit(old["train"]["acc"])  # old-world train only
    cfg_pre = TrainConfig(hidden, lr, batch_size, pretrain_epochs or epochs, seed)
    cfg_ad = TrainConfig(hidden, lr, batch_size, epochs, seed)
    pre, pre_hist = train_mlp(old["train"]["cur"], old["train"]["acc"], old["val"]["cur"], old["val"]["acc"],
                              cfg_pre, xn, yn)  # baseline policy: final-epoch weights
    pre_fp = params_fingerprint(pre)
    lin_old = LinearRegression().fit(old["train"]["cur"], old["train"]["acc"])

    unch = {"mlp": {"new_test": mlp_scores(pre, new["test"], xn, yn), "old_test": mlp_scores(pre, old["test"], xn, yn)},
            "linear": {"new_test": score(new["test"], lin_old.predict(new["test"]["cur"])),
                       "old_test": score(old["test"], lin_old.predict(old["test"]["cur"]))}}
    rec = {"seed": seed, "data_seeds": plan, "k_old": K_OLD, "k_new": K_NEW,
           "pretrain": {"epochs": cfg_pre.epochs, "steps": pre_hist["steps"], "seed": seed,
                        "n_traj": sizes["train"], "n_trans": old["train"]["cur"].shape[0],
                        "checkpoint_fingerprint": pre_fp},
           "unchanged": unch,
           "oracle": {"new_test": score(new["test"], force_for(K_NEW)(new["test"]["cur"]))},
           "budgets": {}}

    for b in budgets:
        d = budget_transitions(new["train"], b)
        entry = {"n_traj": b, "n_trans": d["cur"].shape[0]}
        for name, init in (("finetune", pre), ("fresh", None)):
            m, h = train_mlp(d["cur"], d["acc"], new["val"]["cur"], new["val"]["acc"], cfg_ad, xn, yn,
                             init_model=init, select_best_val=True)
            nt, ot = mlp_scores(m, new["test"], xn, yn), mlp_scores(m, old["test"], xn, yn)
            before, after = unch["mlp"]["old_test"]["accel_mse"], ot["accel_mse"]
            entry[name] = {"new_test": nt, "old_test": ot, "improvement_vs_unchanged_mlp": improvement(unch["mlp"]["new_test"], nt),
                           "old_world_forgetting": {"before_mse": before, "after_mse": after, "abs_increase": after - before,
                                                    "ratio": None if before <= 1e-12 else after / before},
                           "seed": seed, "epochs": cfg_ad.epochs, "steps": h["steps"], "best_epoch": h["best_epoch"],
                           "best_val_loss_norm": min(h["val_loss"]), "init_fingerprint": h["init_fingerprint"]}
        lin = LinearRegression().fit(d["cur"], d["acc"])
        lnt = score(new["test"], lin.predict(new["test"]["cur"]))
        entry["linear_refit"] = {"new_test": lnt, "old_test": score(old["test"], lin.predict(old["test"]["cur"])),
                                 "improvement_vs_unchanged_linear": improvement(unch["linear"]["new_test"], lnt),
                                 "coef": lin.coef_.tolist(), "intercept": lin.intercept_.tolist()}
        rec["budgets"][str(b)] = entry
    return rec


# ---------- aggregation / io ----------
def aggregate(runs):
    """Recursive mean / sample-std (ddof=1) over seeds for numeric leaves; None if any value is None."""
    first = runs[0]
    if isinstance(first, dict):
        return {k: aggregate([r[k] for r in runs]) for k in first}
    if isinstance(first, (list, str)):
        return first if all(r == first for r in runs) else None
    if any(r is None for r in runs):
        return None
    a = np.array(runs, dtype=float)
    return {"mean": float(a.mean()), "std": float(a.std(ddof=1)) if len(a) > 1 else None}


def aggregate_runs(runs):
    keys = ("pretrain", "unchanged", "oracle", "budgets")
    return {k: aggregate([r[k] for r in runs]) for k in keys}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--out", default=os.path.join("outputs", "adaptation"))
    p.add_argument("--smoke", action="store_true", help="1 seed, 3 epochs, separate output dir")
    a = p.parse_args(argv)
    if a.smoke:
        a.seeds, a.epochs = [0], 3
        if a.out == os.path.join("outputs", "adaptation"):
            a.out = os.path.join("outputs", "adaptation_smoke")
    runs = []
    for s in a.seeds:
        print(f"seed {s} ...", flush=True)
        runs.append(run_seed(s, tuple(a.budgets), a.epochs, None, a.hidden, a.lr, a.batch_size))
    agg = aggregate_runs(runs)
    os.makedirs(a.out, exist_ok=True)
    meta = {"config": {"k_old": K_OLD, "k_new": K_NEW, "center": CENTER, "dt": DT, "n_steps": N_STEPS,
                       "budgets": a.budgets, "split_sizes": SIZES, "epochs": a.epochs, "hidden": a.hidden,
                       "lr": a.lr, "batch_size": a.batch_size, "seeds": a.seeds, "smoke": a.smoke,
                       "normalizer_policy": "original old-world train-fitted normalisers for ALL MLPs",
                       "checkpoint_selection": "lowest new-world VALIDATION loss (adapted/fresh MLPs); pretrain = final epoch",
                       "std": "sample std, ddof=1 across seeds",
                       "improvement": "(unchanged_err - method_err)/unchanged_err; null if unchanged_err<=1e-12"},
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "sklearn": sklearn.__version__, "numpy": np.__version__},
            "per_seed": runs, "aggregate": agg}
    with open(os.path.join(a.out, "metrics.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    from plot_adaptation import plot_all
    plot_all(agg, a.budgets, a.out, n_seeds=len(a.seeds))
    print("saved to", a.out)
    return meta


if __name__ == "__main__":
    main()