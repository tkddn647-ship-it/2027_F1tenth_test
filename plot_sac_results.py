"""Plot SAC eval curves (plain vs connectome) from runs/*/eval/evaluations.npz."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _load(path: Path):
    if not path.exists():
        return None
    d = np.load(path)
    steps = np.asarray(d["timesteps"], dtype=np.float64)
    rews = np.asarray(d["results"], dtype=np.float64)  # (n_evals, n_eps)
    lens = np.asarray(d["ep_lengths"], dtype=np.float64)
    return steps, rews.mean(axis=1), rews.std(axis=1), lens.mean(axis=1), lens.std(axis=1)


def main():
    out = Path("docs/figures")
    out.mkdir(parents=True, exist_ok=True)

    series = {
        "plain MLP SAC": Path("runs/plain_Spielberg/eval/evaluations.npz"),
        "connectome SAC": Path("runs/connectome_Spielberg/eval/evaluations.npz"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=140)
    colors = {"plain MLP SAC": "#1f6feb", "connectome SAC": "#cf222e"}

    for name, path in series.items():
        data = _load(path)
        if data is None:
            print(f"[skip] missing {path}")
            continue
        steps, r_m, r_s, l_m, l_s = data
        c = colors[name]
        axes[0].plot(steps, r_m, color=c, lw=2, label=name)
        axes[0].fill_between(steps, r_m - r_s, r_m + r_s, color=c, alpha=0.15)
        axes[1].plot(steps, l_m, color=c, lw=2, label=name)
        axes[1].fill_between(steps, l_m - l_s, l_m + l_s, color=c, alpha=0.15)
        print(
            f"{name}: last steps={int(steps[-1])} "
            f"reward={r_m[-1]:.1f}±{r_s[-1]:.1f} "
            f"ep_len={l_m[-1]:.1f}±{l_s[-1]:.1f} "
            f"best_len={l_m.max():.1f} @ {int(steps[int(np.argmax(l_m))])}"
        )

    axes[0].set_title("Eval mean reward (deterministic)")
    axes[0].set_xlabel("timesteps")
    axes[0].set_ylabel("mean episode reward")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("Eval mean episode length")
    axes[1].set_xlabel("timesteps")
    axes[1].set_ylabel("mean ep length (~0.1 s / step)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Spielberg · real-car aligned (40 m / 40 Hz / ~10 Hz ctrl)", fontsize=11)
    fig.tight_layout()
    png = out / "sac_ab_eval_curves.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"[plot] saved {png.resolve()}")


if __name__ == "__main__":
    main()
