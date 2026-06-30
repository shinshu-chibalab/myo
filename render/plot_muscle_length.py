import os
import numpy as np
import matplotlib.pyplot as plt

def plot_muscle_length(model_name, sim_steps, plot_names, logs, labels, opt_name):

    steps = np.arange(sim_steps)

    save_dir = os.path.join("results", model_name, "muscle_length")
    os.makedirs(save_dir, exist_ok=True)

    for name in plot_names:

        fig, ax1 = plt.subplots(figsize=(8, 4))

        ax2 = ax1.twinx()

        plt.title(f"Muscle: {name}")

        # 左軸
        ax1.plot(steps, logs[0][name], label=labels[0])  # Length
        ax1.plot(steps, logs[1][name], label=labels[1])  # Target
        ax1.plot(steps, logs[2][name], label=labels[2])  # Exctitation
        ax1.plot(steps, logs[3][name], label=labels[3])  # Activation

        # 右軸
        ax2.plot(steps, logs[4][name], linestyle="--", label=labels[4])  # Velocity

        ax1.set_xlabel("Step")

        ax1.set_ylabel("Length / Activation")
        ax2.set_ylabel("Velocity")

        # 凡例統合
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

        ax1.grid(True)

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{opt_name}_{name}.png")

        plt.savefig(save_path)

        plt.close()