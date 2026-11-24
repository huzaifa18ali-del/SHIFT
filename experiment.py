"""End-to-end baseline experiment: linear regression vs MLP on the restoring force.

Usage: python experiment.py [--out outputs/baseline] [--epochs 100] ...
"""
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, sklearn
from sklearn.linear_model import LinearRegression

from simulator import restoring_force, generate_dataset
from train import TrainConfig, Normalizer, train_mlp, predict_mlp, set_seed
from evaluate import evaluate

SEEDS = {"train": 0, "val": 1, "test": 2}          # data-generation seeds (distinct)
SIZES = {"train": 128, "val": 32, "test": 32}      # whole trajectories per split
N_STEPS, DT = 200, 0.01
PHYS = {"k": 1.0, "center_x": 0.0, "center_y": 0.0}


def make_splits():
    f = restoring_force(**PHYS)
    out = {}
    for s in SEEDS:
        traj, (cur, nxt, acc) = generate_dataset(f, SIZES[s], N_STEPS, DT, SEEDS[s])
        out[s] = {"traj": traj, "cur": cur, "nxt": nxt, "acc": acc}
    return out


def run_experiment(cfg, out_dir):
    set_seed(cfg.seed)
    d = make_splits()
    # trajectories never shared across splits: seeds differ => initial states differ
    assert not np.array_equal(d["train"]["traj"][0, :32], d["val"]["traj"][0]), "split overlap"
    assert not np.array_equal(d["train"]["traj"][0, :32], d["test"]["traj"][0]), "split overlap"
    assert not np.array_equal(d["val"]["traj"][0], d["test"]["traj"][0]), "split overlap"

    x_norm = Normalizer().fit(d["train"]["cur"])       # train statistics only
    y_norm = Normalizer().fit(d["train"]["acc"])

    lin = LinearRegression().fit(d["train"]["cur"], d["train"]["acc"])   # raw units, train only
    model, hist = train_mlp(d["train"]["cur"], d["train"]["acc"],
                            d["val"]["cur"], d["val"]["acc"], cfg, x_norm, y_norm)

    te = d["test"]; f = restoring_force(**PHYS)
    preds = {
        "linear_regression": lin.predict(te["cur"]),
        "mlp": predict_mlp(model, te["cur"], x_norm, y_norm, cfg.device),
        "analytical_oracle (simulator reference, not a learned model)": f(te["cur"]),
    }
    metrics = {}
    for name, p in preds.items():
        metrics[name] = {"test": evaluate(te["cur"], te["nxt"], te["acc"], p, DT)}
    for name, p in (("linear_regression", lin.predict(d["val"]["cur"])),
                    ("mlp", predict_mlp(model, d["val"]["cur"], x_norm, y_norm, cfg.device))):
        metrics[name]["val"] = evaluate(d["val"]["cur"], d["val"]["nxt"], d["val"]["acc"], p, DT)
    metrics["linear_coefficients"] = {"coef": lin.coef_.tolist(), "intercept": lin.intercept_.tolist()}

    record = {
        "data_seeds": SEEDS, "trajectories_per_split": SIZES, "n_steps": N_STEPS, "dt": DT,
        "physics": PHYS, "mlp_config": cfg.to_dict(),
        "versions": {"torch": torch.__version__, "sklearn": sklearn.__version__, "numpy": np.__version__},
        "final_train_loss_normalised": hist["train_loss"][-1],
        "final_val_loss_normalised": hist["val_loss"][-1],
        "metrics": metrics,
    }
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as fh:
        json.dump(record, fh, indent=2)

    plt.figure(figsize=(6, 4))
    ep = np.arange(1, len(hist["train_loss"]) + 1)
    plt.plot(ep, hist["train_loss"], label="train"); plt.plot(ep, hist["val_loss"], label="validation")
    plt.yscale("log"); plt.xlabel("epoch"); plt.ylabel("MSE (normalised acceleration)")
    plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(out_dir, "training_curves.png"), dpi=120); plt.close()

    rng = np.random.default_rng(0)
    idx = rng.choice(len(te["acc"]), size=min(2000, len(te["acc"])), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for j, comp in enumerate(["ax", "ay"]):
        ax = axes[j]
        ax.scatter(te["acc"][idx, j], preds["linear_regression"][idx, j], s=6, alpha=0.5, label="linear regression")
        ax.scatter(te["acc"][idx, j], preds["mlp"][idx, j], s=6, alpha=0.5, label="MLP")
        lim = [te["acc"][:, j].min(), te["acc"][:, j].max()]
        ax.plot(lim, lim, "k--", lw=1, label="ideal (y = x)")
        ax.set_xlabel(f"true {comp}"); ax.set_ylabel(f"predicted {comp}"); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, "prediction_comparison.png"), dpi=120); plt.close()
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join("outputs", "baseline"))
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    cfg = TrainConfig(a.hidden, a.lr, a.batch_size, a.epochs, a.seed, a.device)
    rec = run_experiment(cfg, a.out)
    print(json.dumps(rec["metrics"], indent=2))