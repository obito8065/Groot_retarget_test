#!/usr/bin/env python3
"""Read retarget txt, plot L/R finger q1~q6 vs time step (6x2 subplots)."""

import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Column indices: L_finger_q1~q6 = 8~13, R_finger_q1~q6 = 20~25
L_FINGER_COLS = list(range(8, 14))   # q1~q6
R_FINGER_COLS = list(range(20, 26))  # q1~q6

"""
脚本功能：
    1. 读取retarget action txt文件，绘制L/R finger q1~q6 量化数据曲线

执行命令
python eval_plot_retarget_action_fingers.py   \
     -i /vla/users/lijiayi/unifytip_groot/output_video_record/retargeted_actions_20260310_175516.txt \
    -o /vla/users/lijiayi/unifytip_groot/output_video_record/finger_q_plot.png

"""


def load_retarget_data(filepath: str):
    """Read retarget txt, return (time_steps, L_finger_data, R_finger_data)."""
    time_steps = []
    l_fingers = [[] for _ in range(6)]  # q1~q6
    r_fingers = [[] for _ in range(6)]

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) <= 25:
                continue
            time_steps.append(len(time_steps))
            for i in range(6):
                l_fingers[i].append(float(parts[L_FINGER_COLS[i]]))
                r_fingers[i].append(float(parts[R_FINGER_COLS[i]]))

    return (
        np.array(time_steps),
        [np.array(v) for v in l_fingers],
        [np.array(v) for v in r_fingers],
    )


def main():
    parser = argparse.ArgumentParser(description="Plot L/R finger q1~q6 vs time step")
    parser.add_argument(
        "--input",
        "-i",
        default="retargeted_actions_20260310_173553.txt",
        help="Path to retarget txt file",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="finger_q_plot.png",
        help="Path to output image",
    )
    args = parser.parse_args()

    time_steps, l_fingers, r_fingers = load_retarget_data(args.input)

    fig, axes = plt.subplots(6, 2, figsize=(12, 14), sharex=True)
    fig.suptitle("L/R Finger q1~q6 vs Time Step", fontsize=14)

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
