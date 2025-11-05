from pd_gain_optimizer import PDGainOptimizer
import matplotlib.pyplot as plt
import numpy as np
import os

def balance_cost(data, com_init_height):
    # COM(重心)　z
    com_z = np.array(data.subtree_com[0])[2]

    height_cost = (com_init_height - com_z) ** 2

    # energy_cost = np.sum(u**2)

    # return height_cost + 0.01 * energy_cost

    return height_cost


def plot_min_cost_history(optimizer):
    """世代ごとの平均コストを対数スケールで保存のみ"""
    mean_costs = optimizer.cost_history
    generations = range(len(mean_costs))

    plt.figure(figsize=(8,5))
    plt.plot(generations, mean_costs, color='blue', marker='o', markersize=3)
    plt.xlabel("Generation")
    plt.ylabel("Mean Cost")
    plt.title("CMA-ES Mean Cost History")
    plt.yscale("log")  # 縦軸を対数表示
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    # 保存先ディレクトリを作成
    save_dir = os.path.join("results", "plot")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{optimizer.model_name}_history.png")
    plt.savefig(save_path)
    plt.close()  # 画面表示せずに閉じる
    print(f"📌 コスト履歴を保存しました: {save_path}")

if __name__ == "__main__":
    # --- モデル情報 ---
    model_path = "myo_sim/gait2354/gait2354_simbody_cvt3.xml"
    muscles = [
        "glut_med1_r", "glut_med2_r", "glut_med3_r",
        "bifemlh_r", "bifemsh_r", "sar_r", "add_mag2_r", "tfl_r", "pect_r", "grac_r",
        "glut_max1_r", "glut_max2_r", "glut_max3_r",
        "iliacus_r", "psoas_r", "quad_fem_r", "gem_r", "peri_r",
        "rect_fem_r", "vas_int_r", "med_gas_r", "soleus_r", "tib_post_r", "tib_ant_r",

        "glut_med1_l", "glut_med2_l", "glut_med3_l",
        "bifemlh_l", "bifemsh_l", "sar_l", "add_mag2_l", "tfl_l", "pect_l", "grac_l",
        "glut_max1_l", "glut_max2_l", "glut_max3_l",
        "iliacus_l", "psoas_l", "quad_fem_l", "gem_l", "peri_l",
        "rect_fem_l", "vas_int_l", "med_gas_l", "soleus_l", "tib_post_l", "tib_ant_l",

        "ercspn_r", "ercspn_l",
        "intobl_r", "intobl_l",
        "extobl_r", "extobl_l"
    ]

    x0 = [
        1.13454034, 3.53347471, 2.94085894, 0.94123357, 0.47676528, 0.28922884,
        2.09874275, 0.57201523, 2.49719598, 0.45219656, 0.61928558, 0.02217254,
        2.65014797, 0.67930803, 1.18986919, 1.21899574, 0.35070212, 0.95063373,
        0.69819121, 1.08366224, 0.21895890, 2.75224765, 3.53328992, 2.13073795,
        0.50259831, 1.27838355, 1.87859514, 0.44987875, 0.46618444, 2.02861373,
        1.81887829, 1.08945911, 0.09060557, 0.21695965, 0.87575799, 4.21928829,
        0.36873276, 0.06124299, 0.24740956, 1.94658291, 0.35472962, 0.00087767,
        1.02108511, 1.00106203, 2.07043849, 0.76651110, 0.29550459, 0.61229975,
        0.00959015, 0.31753673, 2.51775233, 0.44315556, 1.77156919, 0.10924869,
        0.13465056, 0.15207778, 0.03801949, 0.49715447, 0.26797398, 0.40761359,
        0.05147479, 0.28903550, 0.00439889, 0.27923525, 0.01432585, 0.45834965,
        0.06912743, 0.21455748, 0.06899036, 0.00756826, 0.17085405, 0.00483515,
        0.01451286, 0.57339463, 0.59402406, 0.55168488, 0.59697231, 0.30314917,
        0.07782686, 0.36924134, 0.25190250
    ]


    optimizer = PDGainOptimizer(model_path, muscles, sim_steps=250, model_name="gait2354_v26", output_dir="results", cost_fn=balance_cost)

    # CMA-ES による最適化（1000世代）
    best_params = optimizer.optimize(x0=None, sigma0=1.0, maxiter=10000, popsize=112, delay_time=0.10, noise_std=0.01, n_jobs=56)
    plot_min_cost_history(optimizer)

    num_muscles = len(muscles)
    Kp_opt = best_params[:num_muscles]
    Kd_opt = best_params[num_muscles:2*num_muscles]
    target_len = best_params[2*num_muscles:]
    print("最適化された Kp:", Kp_opt)
    print("最適化された Kd:", Kd_opt)
    print("最適化された muscle_len:", target_len)


# --- version ---

"""
V10: Wc = 1e2, We = 1e-5, delay_time = 0.00, noise_std = 0.00
V11: Wc = 1e2, We = 1e-5, delay_time = 0.05, noise_std = 0.01
V12: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01
V13: Wc = 1e2, We = 1e-4, delay_time = 0.00, noise_std = 0.00
V14: Wc = 1e2, We = 1e-4, delay_time = 0.05, noise_std = 0.01
V15: Wc = 1e2, We = 1e-4, delay_time = 0.10, noise_std = 0.01
V16 test: Wc = 1e2, We = 1e-4, delay_time = 0.0, noise_std = 0.0, maxtier=1000
V17: Wc = 1e2, We = 1e-4, delay_time = 0.00, noise_std = 0.00, x0=None
V18: Wc = 1e2, We = 1e-5, delay_time = 0.00, noise_std = 0.00, x0=None
V19: Wc = 1e2, We = 1e-5, delay_time = 0.00, noise_std = 0.00, sigmoid: e^(-3x+14)
V20: Wc = 1e2, We = 1e-5, delay_time = 0.00, noise_std = 0.00, Softplus log1p(exp(x-10))
V21: Wc = 1e2, We = 1e-5, delay_time = 0.00, noise_std = 0.00, Softplus log1p(exp(x-5))
V22: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-5)), x0=None
V23: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-5))
V24: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-1))
V25: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-10)), x0=None
V26: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-4)), x0=None
"""