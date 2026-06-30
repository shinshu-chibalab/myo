import os
import matplotlib.pyplot as plt

def plot_cost_history(model_name, log, mode):
    """世代ごとのコストを対数スケールで保存"""

    generations = range(len(log))

    # 保存先ディレクトリを作成
    save_dir = os.path.join("results", model_name, "plot_cost")
    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(8,5))
    plt.plot(generations, log, color='blue')
    plt.xlabel("Generation")
    plt.ylabel("Minimum cost of COM trajectory length and falling time")
    plt.title("best_fitness history")
    # plt.yscale("log") 
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    save_path = os.path.join(save_dir, f"{model_name}_history_{mode}.png")
    plt.savefig(save_path)
    plt.close()  # 画面表示せずに閉じる
    print(f"コスト履歴を保存しました: {save_path}")