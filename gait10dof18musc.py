from pd_gain_optimizer import PDGainOptimizer
from shapely.geometry import Point, Polygon
import numpy as np


def balance_cost(data, sim_steps, com_init_height, bos_polygon=None, min_ratio=0.9, com_w=1.0):
    # COM(重心)　xy, z
    # com_xy = np.array(data.subtree_com[0])[:2]
    com_z = np.array(data.subtree_com[0])[2]

    # if bos_polygon is None:
    #     # BOS が存在しない場合は高コスト
    #     bos_dist = 1e2  # 適宜調整
    # else:
    #     bos_center = np.array(bos_polygon.centroid.coords[0])
    #     bos_dist = np.linalg.norm(com_xy - bos_center)

    # # BOS外がどうか
    # if not bos_polygon.contains(Point(com)):
    #     return sim_steps - data.time

    # # --- COP (簡易: 足接触点の平均) ---
    # contacts = [data.contact[i] for i in range(data.ncon)]
    # if contacts:
    #     pts = [c.pos[:2] for c in contacts]
    #     cop = np.mean(pts, axis=0)
    #     cop_cost = np.linalg.norm(com - cop)**2
    # else:
    #     cop_cost = 100  # 足裏接触なしはペナルティ

    # # 重心高さの評価１
    # threshold = min_ratio * com_init_height
    # if com_z < threshold:
    #     height_cost = (com_init_height - com_z) ** 2
    # else:
    #     height_cost = 0.0

    height_cost = (com_init_height - com_z) ** 2

    # return com_w * bos_dist + height_cost
    return height_cost

if __name__ == "__main__":
    # --- モデル情報 ---
    model_path = "myo_sim/gait10dof18musc/gait10dof18musc_cvt3.xml"
    muscles = ["hamstrings_r","glut_max_r","iliopsoas_r",
               "rect_fem_r","vasti_r","gastroc_r","soleus_r","tib_ant_r",
               "hamstrings_l","glut_max_l","iliopsoas_l",
               "rect_fem_l","vasti_l","gastroc_l","soleus_l","tib_ant_l"]

    optimizer = PDGainOptimizer(model_path, muscles, sim_steps=300, model_name="gait10dof18musc1", output_dir="results", cost_fn=balance_cost)

    x0 = [
        1.45363749e+01, 1.22351785e+01, 6.94043347e+00, 1.24061720e+01, 
        5.25865301e+00, 1.76787331e+00, 1.70833896e+01, 1.80327462e+01, 
        1.11505042e+01, 8.20501483e+00, 1.17082827e+01, 1.47266639e+01, 
        5.83362793e+00, 1.08817607e+01, 1.11830330e+01, 3.58045889e-01, 
        1.49911031e-04, 5.06595068e+00, 9.03261097e+00, 2.79429253e-02, 
        3.87549957e-02, 1.05714014e-03, 1.22272550e+01, 2.74848325e+00, 
        4.64453852e-03, 9.15160518e+00, 4.13759108e+00, 3.93196090e-01, 
        7.61675498e-02, 3.32300480e-01, 1.79798089e-01, 3.38953853e+00
    ]

    # CMA-ES による最適化（1000世代）
    best_params = optimizer.optimize(x0=None, sigma0=2.0, maxiter=1000, delay_time=0.05, noise_std=0.01)
    print("最適化されたパラメータ:", best_params)

    # 最適化されたゲインでレンダリング付きシミュレーション
    optimizer.run_simulation(best_params, render=True, plot=False)

    num_muscles = len(muscles)
    Kp_opt = best_params[:num_muscles]
    Kd_opt = best_params[num_muscles:2*num_muscles]
    print("最適化された Kp:", Kp_opt)
    print("最適化された Kd:", Kd_opt)