import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D  # noqa

def plot_pareto_front(model_name, fronts):
    """
    fronts:
      - [front1]           ← 第1フロントのみ
      - [front1, front2…]  ← 複数フロント
    """

    # --- 保存先 ---
    save_dir = os.path.join("results", model_name, "plot_cost")
    os.makedirs(save_dir, exist_ok=True)

    # --- Figure は1回だけ ---
    plt.figure(figsize=(6, 5))

    for i, pareto_front in enumerate(fronts):
        # --- fitness 取得 ---
        f1 = [ind.fitness.values[0] for ind in pareto_front]
        f2 = [ind.fitness.values[1] for ind in pareto_front]

        # --- f1 昇順でソート ---
        f1, f2 = zip(*sorted(zip(f1, f2)))

        plt.scatter(
            f1, f2,
            s=2,
            alpha=1.0,
            label=f"Front {i+1}"
        )

    plt.xlabel("Energy cost")
    plt.ylabel("COM cost")
    plt.title(f"{model_name} Pareto Front")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if len(fronts) == 1:
        save_path = os.path.join(save_dir, f"{model_name}_front.png")
    else:
        save_path = os.path.join(save_dir, f"{model_name}_fronts.png")

    plt.savefig(save_path)
    plt.close()

    print(f"パレートフロントを保存しました: {save_path}")



def plot_pareto_front_hist(model_name, front_hist):
    """
    front_hist: [(gen, pareto_front), ...]
    """

    save_dir = os.path.join("results", model_name, "plot_cost")
    os.makedirs(save_dir, exist_ok=True)

    gens = [gen for gen, _ in front_hist]
    gen_min, gen_max = min(gens), max(gens)

    norm = Normalize(vmin=gen_min, vmax=gen_max)
    cmap = cm.viridis  # 見やすく論文向け

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for gen, front in front_hist:
        f1 = np.array([ind.fitness.values[0] for ind in front])
        f2 = np.array([ind.fitness.values[1] for ind in front])
        g  = np.full(len(front), gen)

        order = np.argsort(f1)
        f1, f2, g = f1[order], f2[order], g[order]

        colors = cmap(norm(gen))

        ax.scatter(
            f1,
            f2,
            g,
            color=colors,
            s=6,
            alpha=0.6
        )

        ax.plot(
            f1,
            f2,
            g,
            color=colors,
            alpha=0.7,
            linewidth=1.0
        )

    # --- 軸ラベル ---
    ax.set_xlabel("Energy cost")
    ax.set_ylabel("COM cost")
    ax.set_zlabel("Generation")

    ax.view_init(elev=25, azim=135)

    # --- カラーバー（世代） ---
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.1)
    cbar.set_label("Generation")

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_pareto_history.png")
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"3D Pareto front (gradient) saved: {save_path}")

