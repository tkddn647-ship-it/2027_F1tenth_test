"""
viz_policy_layers.py
====================
Connectome SAC 한 스텝의 레이어/커넥톰 활성화를 시각화.

  python viz_policy_layers.py --model connectome_sac_f1tenth_Spielberg_80k.zip --map Spielberg --use-cache --seed 0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import SAC

from connectome_loader import (
    build_adjacency,
    identify_io_neurons,
    load_hemibrain_cache,
    select_speed_relevant_subcircuit,
)
from f1tenth_mapless_env import F1TenthMaplessEnv, N_BEAMS
from policy_sac_connectome import ConnectomeTemporalFeatures, HIST_LEN


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().float().cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--map", type=str, default="Spielberg")
    p.add_argument("--use-cache", action="store_true")
    p.add_argument("--cache-dir", type=str, default="hemibrain_cache")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup", type=int, default=30, help="결정적 주행 후 스냅샷")
    p.add_argument("--min-speed", type=float, default=0.5)
    p.add_argument("--max-speed", type=float, default=7.0)
    p.add_argument("--max-steer", type=float, default=0.30)
    p.add_argument("--out", type=str, default="docs/figures/policy_layers_activation.png")
    args = p.parse_args()

    env = F1TenthMaplessEnv(
        args.map,
        seed=args.seed,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        max_steer=args.max_steer,
    )
    custom = None
    if args.use_cache:
        n, c = load_hemibrain_cache(args.cache_dir)
        n, c = select_speed_relevant_subcircuit(n, c, max_neurons=256)
        n = n.reset_index(drop=True)
        i, o = identify_io_neurons(n)
        A = build_adjacency(n, c)
        custom = {"ConnectomeTemporalFeatures": ConnectomeTemporalFeatures}
        n_in, n_out, n_neu = len(i), len(o), len(n)
    else:
        raise SystemExit("--use-cache 필요 (connectome 정책 시각화)")

    model = SAC.load(args.model, env=env, custom_objects=custom, device="cpu")
    fe = model.policy.actor.features_extractor
    if not hasattr(fe, "forward_intermediates"):
        raise SystemExit("features extractor에 forward_intermediates 없음 — 코드 갱신 필요")

    obs, _ = env.reset(seed=args.seed)
    info = {}
    for _ in range(args.warmup):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            obs, _ = env.reset()
            info = {}

    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        acts = fe.forward_intermediates(obs_t)
        action, _ = model.predict(obs, deterministic=True)

    lidar = _to_np(acts["lidar"])[0]  # (T, 135)
    yaw = _to_np(acts["yaw"])[0]
    z = _to_np(acts["z"])[0]  # (T, 64)
    h_seq = _to_np(acts["h_seq"])[0]  # (T, N)
    y_dn = _to_np(acts["y_dn"])[0]
    skip = _to_np(acts["skip"])[0]
    fuse = _to_np(acts["fuse"])[0]
    h_last = _to_np(acts["h"])[0]

    # topology density for caption
    A_abs = np.abs(A)
    dens = 100.0 * (A_abs > 0).mean()

    fig = plt.figure(figsize=(14, 10), dpi=130)
    fig.suptitle(
        f"Connectome SAC forward @ {args.map}  |  "
        f"neurons={n_neu} in={n_in} DN={n_out} dens={dens:.3f}%  |  "
        f"action steer={action[0]:+.2f} speed_u={action[1]:.2f}  "
        f"v={info.get('speed', env.v):.1f} slip={info.get('slip_angle', 0):+.3f}",
        fontsize=11,
    )

    # 1) pipeline text
    ax0 = fig.add_subplot(3, 3, 1)
    ax0.axis("off")
    ax0.text(
        0.0,
        0.95,
        "Forward pass (one control step)\n"
        "─────────────────────────────\n"
        "1. Obs: LiDAR 135x5 + yaw x5\n"
        "2. LidarEnc(Conv1d) -> z_L 48\n"
        "   IMUEnc(MLP)      -> z_I 16\n"
        "3. Unroll t=0..4:\n"
        "   z_t=[z_L;z_I] -> ConnectomeRNN(h)\n"
        "4. DN y_4  ||  skip(mean z)\n"
        "5. Fuse -> 256 features\n"
        "6. SAC Actor -> steer, speed\n"
        "\n"
        "Fixed: A_signed topology\n"
        "Learned: scale, W_in, enc,\n"
        "         skip, fuse, pi/Q",
        va="top",
        family="DejaVu Sans Mono",
        fontsize=8.5,
        transform=ax0.transAxes,
    )
    ax0.set_title("Pipeline")

    # 2) latest lidar polar-ish as 1d
    ax1 = fig.add_subplot(3, 3, 2)
    angles = np.linspace(-np.deg2rad(135), np.deg2rad(135), N_BEAMS)
    ax1.plot(np.rad2deg(angles), lidar[-1] * 40.0, color="#1f6feb", lw=1.2)
    ax1.set_xlabel("beam angle (deg)")
    ax1.set_ylabel("range (m)")
    ax1.set_title("LiDAR (newest frame)")
    ax1.grid(True, alpha=0.3)

    # 3) lidar history heatmap
    ax2 = fig.add_subplot(3, 3, 3)
    im = ax2.imshow(lidar, aspect="auto", cmap="viridis", origin="lower")
    ax2.set_xlabel("beam index")
    ax2.set_ylabel("hist t (0=old)")
    ax2.set_title("LiDAR history (norm)")
    fig.colorbar(im, ax=ax2, fraction=0.046)

    # 4) encoder z over time
    ax3 = fig.add_subplot(3, 3, 4)
    ax3.imshow(z.T, aspect="auto", cmap="coolwarm", origin="lower")
    ax3.set_xlabel("hist t")
    ax3.set_ylabel("z dim (lidar+imu)")
    ax3.set_title("Encoder z_t  (64)")
    ax3.axhline(47.5, color="k", lw=0.8, ls="--")  # lidar|imu split ~48

    # 5) yaw hist
    ax4 = fig.add_subplot(3, 3, 5)
    ax4.bar(np.arange(HIST_LEN), yaw, color="#cf222e", alpha=0.8)
    ax4.set_xlabel("hist t")
    ax4.set_ylabel("yaw rate (norm)")
    ax4.set_title("IMU yaw history")
    ax4.grid(True, alpha=0.3)

    # 6) connectome hidden energy over time
    ax5 = fig.add_subplot(3, 3, 6)
    energy = np.linalg.norm(h_seq, axis=1)
    ax5.plot(np.arange(HIST_LEN), energy, "o-", color="#8250df")
    ax5.set_xlabel("unroll t")
    ax5.set_ylabel("||h_t||")
    ax5.set_title("ConnectomeRNN hidden energy")
    ax5.grid(True, alpha=0.3)

    # 7) DN readout
    ax6 = fig.add_subplot(3, 3, 7)
    ax6.bar(np.arange(len(y_dn)), y_dn, color="#1a7f37", alpha=0.85)
    ax6.set_xlabel("DN index")
    ax6.set_ylabel("activation")
    ax6.set_title(f"DN output y_4  (n={len(y_dn)})")
    ax6.grid(True, alpha=0.3)

    # 8) skip vs fuse
    ax7 = fig.add_subplot(3, 3, 8)
    ax7.plot(skip, label="skip(128)", color="#0969da", alpha=0.8, lw=1)
    ax7.plot(fuse[:128], label="fuse[:128]", color="#cf222e", alpha=0.7, lw=1)
    ax7.set_title("skip ‖ fuse (first 128)")
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)

    # 9) connectome population snapshot
    ax8 = fig.add_subplot(3, 3, 9)
    order = np.argsort(np.abs(h_last))[::-1]
    top = order[:80]
    ax8.bar(np.arange(len(top)), h_last[top], color="#8250df", alpha=0.85)
    ax8.set_xlabel("neuron (top-|h|)")
    ax8.set_ylabel("h")
    ax8.set_title("Connectome h (top 80 |activation|)")
    ax8.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    print(f"[viz] saved {out.resolve()}")
    print(
        f"[viz] ||h||={np.linalg.norm(h_last):.3f}  "
        f"DN mean={y_dn.mean():+.3f}  fuse||={np.linalg.norm(fuse):.2f}"
    )


if __name__ == "__main__":
    main()
