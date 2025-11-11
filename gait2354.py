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
        4.14516382e-02, 2.11612879e+00, 1.21279213e+00, 7.44526076e+00,
        4.56988357e+00, 4.23316473e+00, 1.14608853e+01, 3.07291561e+00,
        6.12597757e-05, 5.51103301e+00, 4.70817079e+00, 1.67970939e+00,
        3.85714062e+00, 6.95167947e+00, 4.37199495e+00, 1.42890586e+00,
        3.08725775e+00, 2.84052617e+00, 1.15430343e+01, 6.25659012e+00,
        1.04665250e+01, 1.31384494e+01, 3.51876107e+00, 8.41778250e+00,
        4.90027581e+00, 3.40277123e+00, 2.62311166e+00, 1.16404734e+00,
        6.28618673e+00, 9.37540074e+00, 4.86597432e-01, 2.03012723e+00,
        1.37088452e+01, 4.95765951e+00, 7.77885961e+00, 7.48612822e+00,
        1.99823323e+00, 3.09549383e+00, 6.09044275e+00, 5.92643357e+00,
        5.10288144e-01, 1.21553231e+00, 1.42713058e+01, 8.93896888e+00,
        7.08600223e+00, 1.50315382e-01, 1.27153925e-03, 1.68674480e+00,
        1.40049206e+00, 4.19185779e+00, 4.96818108e-01, 5.50684351e+00,
        6.48990216e+00, 6.84106316e+00, 9.60918492e-03, 4.82778937e-01,
        2.08186448e-04, 5.29100983e-01, 5.99468391e-01, 5.99526962e-01,
        1.22418495e-03, 5.75119871e-01, 1.25711924e-02, 5.68737784e-01,
        5.83818627e-01, 2.21788984e-01, 7.39551566e-04, 4.66120586e-01,
        5.35707271e-01, 3.12352754e-01, 2.50665656e-01, 1.13610096e-01,
        5.69110707e-01, 5.99803128e-01, 5.89405640e-01, 5.84292758e-01,
        5.98718757e-01, 5.92883066e-01, 5.65236139e-01, 3.33999819e-02,
        3.14453923e-01
    ]



    optimizer = PDGainOptimizer(model_path, muscles, sim_steps=500, model_name="gait2354_v39", output_dir="results", cost_fn=balance_cost)

    # CMA-ES による最適化（10,000世代）
    best_params = optimizer.optimize(x0=x0, sigma0=1.0, maxiter=10000, popsize=112, delay_time=0.10, noise_std=0.05, n_jobs=56)

    # PSOによる最適化 （10,000世代）
    # best_params = optimizer.pso(x0=None, sigma0=2.0, maxiter=10000, popsize=112, delay_time=0.10, noise_std=0.01, n_jobs=56)
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
V27: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-4)), x0=None, optimizer=PSO
V28: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.05, Softplus log1p(exp(x-4)), x0=None
V29: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.10, Softplus log1p(exp(x-4)), x0=None

step=250
V30: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-30)), x0=None
step=500
V30: Wc = 1e2, We = 1e-5, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-30)), x0=None
V35: Wc = 1.0, We = 3e-8, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075))
V36: Wc = 1.0, We = 1e-7, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075))
V37: Wc = 1.0, We = 1e-6, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075))
V38: Wc = 1.0, We = 5e-6, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075))
V39: Wc = 1.0, We = 5e-6, delay_time = 0.10, noise_std = 0.05, Softplus log1p(exp(x-0.0075))
"""