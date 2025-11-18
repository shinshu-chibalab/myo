from pd_gain_optimizer import PDGainOptimizer
from pd_gain_optimizer1 import PDGainOptimizer1
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


def plot_min_cost_history(opt_name, optimizer):
    """世代ごとの平均コストを対数スケールで保存のみ"""
    min_costs = optimizer.cost_history
    generations = range(len(min_costs))

    plt.figure(figsize=(8,5))
    plt.plot(generations, min_costs, color='blue', marker='o', markersize=3)
    plt.xlabel("Generation")
    plt.ylabel("Min Cost")
    plt.title(f"{opt_name} Min Cost History")
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
    # model_path = "myo_sim/gait10dof18musc/gait10dof18musc_fixed.xml"
    # muscles =[
    #     "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r", "gastroc_r", "soleus_r", "tib_ant_r",
    #     "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l", "gastroc_l", "soleus_l", "tib_ant_l",
    #     ]
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

    x0 = np.array([
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
    ])

    # x0 = np.array([
    #     1.18200664, 0.84469548, 2.80531174, 6.05283853, 7.82001703, 4.16273122,
    #     10.60792671, 1.65164375, 2.20832162, 3.02492703, 5.85840604, 2.99132799,
    #     2.30469697, 5.55714757, 5.07588459, 4.43903058, 5.17849085, 1.50297311,
    #     10.88296957, 6.55926094, 10.24934447, 12.53643706, 3.29801609, 10.13522276,
    #     1.86871039, 3.30358442, 0.79395662,
    #     4.20061842, 6.95705485, 9.99109103, 0.4403686, 2.54311686, 10.65962007,
    #     7.06626869, 6.97958551, 8.71940371, 4.73555346, 7.07276413, 6.40708416,
    #     7.63611323, 4.37400651, 0.41854603, 14.92704337, 8.08112058, 8.70737698,
    #     0.57271943, 0.63501919, 0.43702113, 0.60111769, 3.95699597, 0.18647659,
    #     4.79323454, 7.66338979, 6.24319208,
    #     5.92343676e-01, 1.06316693e-01, 4.80300436e-03, 5.14314362e-01,
    #     4.78678182e-01, 5.99961565e-01, 5.46817378e-02, 5.93337782e-01,
    #     3.68627753e-01, 5.94105060e-01, 6.46338029e-02, 5.99994215e-01,
    #     3.37787950e-01, 5.92362755e-01, 5.30890031e-01, 1.59677491e-02,
    #     9.79009540e-03, 5.47602228e-04, 5.00742452e-01, 3.74689100e-01,
    #     5.97333586e-01, 4.25083594e-01, 4.28927186e-01, 4.47808352e-01,
    #     5.27022001e-01, 2.72283265e-04, 3.93580765e-01
    # ])

    # x0 = np.array([
    #     8.45075581e+00, 5.81407248e+00, 6.37050613e+00, 6.17471594e+00,
    #     9.99999086e+00, 9.99999678e+00, 5.58284523e+00, 9.96879015e+00,
    #     3.14688045e+00, 9.96338496e+00, 8.90394188e+00, 7.58141459e+00,
    #     9.99835607e+00, 8.20710179e+00, 9.62044657e+00, 9.99999975e+00,
    #     3.31280913e-06, 9.68562798e+00, 4.04422120e+00, 7.75226702e+00,
    #     5.06278657e+00, 9.03925878e+00, 4.92815878e+00, 6.72889273e+00,
    #     9.07368935e+00, 1.76777083e+00, 1.02564438e+00,
    #     2.09002405e+00, 8.60893931e+00, 9.98935174e+00, 1.69890884e+00,
    #     8.22833086e-06, 3.49824275e-07, 9.99999796e+00, 9.34502558e-01,
    #     6.02991166e-02, 4.66075684e+00, 2.67327525e+00, 3.18781374e-01,
    #     8.81140392e+00, 2.80684823e-01, 2.93745121e-02, 9.99980277e+00,
    #     9.99999790e+00, 9.38861338e+00, 2.27407492e-01, 4.02759210e-02,
    #     5.15705502e-01, 8.55182141e-03, 2.43557699e+00, 6.21571720e+00,
    #     5.88685977e-01, 5.14913352e-01, 2.56661581e+00,
    #     0.1679569,  0.17056279, 0.11743175, 0.4671169,  0.249268,
    #     0.58150099, 0.1655107,  0.57707847, 0.13458946, 0.57117925,
    #     0.21905537, 0.24522751, 0.24475037, 0.2288658,  0.28679666,
    #     0.00903833, 0.04947981, 0.10703882, 0.50097238, 0.24700466,
    #     0.49212503, 0.31355441, 0.36828423, 0.31757455, 0.19086388,
    #     0.21028364, 0.32955408
    # ])

    sim_steps=500
    model_name="gait2354_v49"
    output_dir="results"
    cost_fn=balance_cost

    sigma=0.5
    maxiter=10000
    popsize=810

    delay_time=0.10
    noise_std=0.01

    n_jobs=56

    # optimizer = PDGainOptimizer(model_path, muscles, sim_steps=sim_steps, model_name=model_name, output_dir=output_dir, cost_fn=cost_fn)
    optimizer = PDGainOptimizer1(model_path, muscles, sim_steps=sim_steps, model_name=model_name, output_dir=output_dir, cost_fn=cost_fn)

    # CMA-ES による最適化（10,000世代）
    # best_params = optimizer.optimize(x0=None, sigma0=sigma, maxiter=maxiter, popsize=popsize, delay_time=delay_time, noise_std=noise_std, n_jobs=n_jobs)
    # plot_min_cost_history("CMA-ES", optimizer)

    # PSOによる最適化 （10,000世代）
    best_params = optimizer.pso(x0=x0, sigma0=sigma, maxiter=maxiter, popsize=popsize, delay_time=delay_time, noise_std=noise_std, n_jobs=n_jobs)
    plot_min_cost_history("PSO", optimizer)

    optimizer.render_video(best_params, camera="front", delay_time=delay_time, noise_std=delay_time)
    optimizer.render_video(best_params, camera="diagonal", delay_time=delay_time, noise_std=delay_time)
    optimizer.render_video(best_params, camera="side", delay_time=delay_time, noise_std=delay_time)
    optimizer.render_video(best_params, camera="oblique", delay_time=delay_time, noise_std=delay_time)

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
V40: Wc = 1.0, We = 5e-6, delay_time = 0.10, noise_std = 0.05, Softplus log1p(exp(x-0.0075))(もう一度)
V41: Wc = 1.0, We = 1e-5, delay_time = 0.10, noise_std = 0.05, Softplus log1p(exp(x-0.0075))

step = 500
V42: Wc = 1.0, We = 1e-6, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075) pso
V47: Wc = 1.0, We = 5e-6, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075) pso x0=x0
V48: Wc = 1.0, We = 1e-6, delay_time = 0.10, noise_std = 0.01, Softplus log1p(exp(x-0.0075) x0=None
V48: Wc = 1.0, We = 1e-7, delay_time = 0.10, noise_std = 0.01, pso total_cost = com_cost(com_log) + 1e-7 * total_Edot
"""