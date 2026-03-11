#!/usr/bin/env python3
"""Read parquet file, plot L/R finger q1~q6 vs time step (6x2 subplots)."""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# GR1ArmsAndWaistFourierHands action layout (from modality.json):
# left_hand:  [7:13]  -> L_finger_q1~q6
# right_hand: [29:35] -> R_finger_q1~q6
L_FINGER_START, L_FINGER_END = 7, 13
R_FINGER_START, R_FINGER_END = 29, 35


def load_parquet_fingers(filepath: str):
    """Read parquet, return (time_steps, L_finger_data, R_finger_data)."""
    df = pd.read_parquet(filepath)
    actions = np.stack(df["action"].apply(lambda x: np.array(x, dtype=np.float32)))
    n = len(actions)
    time_steps = np.arange(n)
    l_fingers = [actions[:, i] for i in range(L_FINGER_START, L_FINGER_END)]
    r_fingers = [actions[:, i] for i in range(R_FINGER_START, R_FINGER_END)]
    return time_steps, l_fingers, r_fingers


def main():
    parser = argparse.ArgumentParser(description="Plot L/R finger q1~q6 vs time step from parquet")
    parser.add_argument(
        "--input",
        "-i",
        default="/vla/users/lijiayi/robocasa_datasets_full/pick_and_place_lerobot_task24_sampled_300/gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_300/data/chunk-000/episode_000000.parquet",
        help="Path to parquet file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="finger_q_parquet_plot.png",
        help="Path to output image",
    )
    args = parser.parse_args()

    time_steps, l_fingers, r_fingers = load_parquet_fingers(args.input)

    fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
    fig.suptitle("L/R Finger q1~q6 vs Time Step (from parquet)", fontsize=14)

    for i in range(6):
        axes[i, 0].plot(time_steps, l_fingers[i], linewidth=0.8, color="#2563eb")
        axes[i, 0].set_ylabel(f"L_finger_q{i+1} (rad)", fontsize=10)
        axes[i, 0].grid(True, alpha=0.3)

        axes[i, 1].plot(time_steps, r_fingers[i], linewidth=0.8, color="#dc2626")
        axes[i, 1].set_ylabel(f"R_finger_q{i+1} (rad)", fontsize=10)
        axes[i, 1].grid(True, alpha=0.3)

    axes[5, 0].set_xlabel("Time Step", fontsize=10)
    axes[5, 1].set_xlabel("Time Step", fontsize=10)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    plt.close()

    print(f"Saved to: {args.output}")
    print(f"Data points: {len(time_steps)}")


if __name__ == "__main__":
    main()
