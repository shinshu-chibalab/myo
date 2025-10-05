from pd_gain_optimizer import PDGainOptimizer
from shapely.geometry import Point, Polygon
import numpy as np


def balance_cost(data, diff_len, v, u, step, bos_polygon, muscle_cost=1.0, cop_weight=5.0):
    # 筋長誤差
    muscle_cost = np.sum(diff_len**2)

    # COM(重心)　x, y のみ
    com = np.array(data.subtree_com[0])[:2]
    if not bos_polygon.contains(Point(com)):
        raise RuntimeError("FALL")  # シミュレーション強制終了

    # --- COP (簡易: 足接触点の平均) ---
    contacts = [data.contact[i] for i in range(data.ncon)]
    if contacts:
        pts = [c.pos[:2] for c in contacts]
        cop = np.mean(pts, axis=0)
        cop_cost = np.linalg.norm(com - cop)**2
    else:
        cop_cost = 1e3  # 足裏接触なしはペナルティ

    return muscle_w * muscle_cost + cop_w * cop_cost

if __name__ == "__main__":
    # --- モデル情報 ---
    model_path = "myo_sim/gait10dof18musc/gait10dof18musc_cvt3.xml"
    muscles = ["hamstrings_r","glut_max_r","iliopsoas_r",
               "rect_fem_r","vasti_r","gastroc_r","soleus_r","tib_ant_r",
               "hamstrings_l","glut_max_l","iliopsoas_l",
               "rect_fem_l","vasti_l","gastroc_l","soleus_l","tib_ant_l"]

    optimizer = PDGainOptimizer(model_path, muscles, model_name="gait10dof18musc_test2", sim_steps=150)

    # CMA-ES による最適化（1000世代）
    best_params = optimizer.optimize(maxiter=1000, delay_time=0.05, noise_std=0.01)
    print("最適化されたパラメータ:", best_params)

    # 最適化されたゲインでレンダリング付きシミュレーション
    optimizer.run_simulation(best_params, render=True, plot=True)

    num_muscles = len(muscles)
    Kp_opt = best_params[:num_muscles]
    Kd_opt = best_params[num_muscles:]
    print("最適化された Kp:", Kp_opt)
    print("最適化された Kd:", Kd_opt)