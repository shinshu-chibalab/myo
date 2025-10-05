import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import skvideo.io
from mujoco import MjModel, MjData, mj_step, Renderer, mj_forward
from scipy.spatial import ConvexHull
from cma import CMAEvolutionStrategy
from shapely.geometry import Point, Polygon


class PDGainOptimizer:
    def __init__(self, model_path, muscles, sim_steps=100,
                 model_name=None, output_dir="results", cost_fn=None):
        self.model = MjModel.from_xml_path(model_path)
        self.muscles = muscles
        self.tendon_ids = [self.model.tendon(f"{m}_tendon").id for m in muscles]
        self.actuator_ids = [self.model.actuator(m).id for m in muscles]

        data = MjData(self.model)
        data.qpos[:] = self.model.key_qpos[0].copy()
        data.qvel[:] = 0
        data.qacc[:] = 0
        mj_forward(self.model, data)
        self.com_init_height = np.array(data.subtree_com[0])[2]

        self.init_len = np.array([data.ten_length[t_id] for t_id in self.tendon_ids])
        print(self.init_len)

        # if target_len is None:
        #     self.target_len = np.array([data.ten_length[t_id] for t_id in self.tendon_ids])
        # else:
        #     self.target_len = np.array(target_len)

        self.sim_steps = sim_steps
        self.model_name = model_name if model_name else os.path.splitext(os.path.basename(model_path))[0]
        self.output_dir = output_dir

        com_init = np.array(data.subtree_com[0])[:2]

        # 評価関数（デフォルト: 筋長誤差の二乗和）
        if cost_fn is None:
            self.cost_fn = lambda data, diff_len, v, u, step: np.sum(diff_len**2)
        else:
            self.cost_fn = cost_fn

        # 出力ディレクトリ作成
        self.video_dir = os.path.join(output_dir, "videos")
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.bos_active = False

    def run_simulation(self, params, render=False, plot=False, delay_time=0.01, noise_std=0.001):
        data = MjData(self.model)
        num_muscles = len(self.muscles)

        # パラメータを3分割[Kp, Kd, target_len]
        Kp = np.array(params[:num_muscles])
        Kd = np.array(params[num_muscles:2*num_muscles])
        target_len = np.array(params[2*num_muscles:])
        total_cost = 0.0
        
        data.qpos[:] = self.model.key_qpos[0]
        data.qvel[:] = 0
        data.qacc[:] = 0

        # シミュレーション刻み時間
        dt = self.model.opt.timestep
        delay_steps = max(1, int(delay_time / dt))

        # 遅延用バッファ
        length_buffer = []
        velocity_buffer = []

        time_log = []
        length_log = [[] for _ in range(num_muscles)]

        cop_log = []

        baseline = 0.05
        reflex_gain = 2.0

        renderer = Renderer(self.model, height=400, width=400) if render else None
        frames = []

        for step in range(self.sim_steps):
            l = data.ten_length[self.tendon_ids]
            v = data.ten_velocity[self.tendon_ids]

            # bos_polygon = compute_bos_polygon(self.model, data)

            # com_xy = np.array(data.subtree_com[0])[:2]

            # if not self.bos_active:
            #     # まだ接触していない
            #     if bos_polygon is not None:
            #         self.bos_active = True
            #     cost = 0

            # else:
            #     # すでに接触後
            #     if bos_polygon is None:
            #         # 支持がなくなった → 高コスト
            #         cost = 1e2  # 適当に大きな値
            #     else:
            #         # BOS が存在するとき
            #         if bos_polygon.contains(Point(com_xy)):
            #             cost = 0
            #         else:
            #             bos_center = np.mean(np.array(bos_polygon.exterior.coords), axis=0)
            #             cost = np.linalg.norm(com_xy - bos_center)


            # total_cost += cost

            # バッファに保存
            length_buffer.append(l.copy())
            velocity_buffer.append(v.copy())

            # 遅延がある場合
            if len(length_buffer) > delay_steps:
                l_delayed = length_buffer[-delay_steps]
                v_delayed = velocity_buffer[-delay_steps]
            else:
                l_delayed = l
                v_delayed = v

            diff_len = l_delayed - target_len            

            reflex = reflex_gain * np.clip(diff_len, 0, None)
            u = baseline + Kp * diff_len + Kd * v + reflex

            # ノイズを加える
            noise = np.random.normal(0, noise_std, size=u.shape)
            u += noise

            u = np.clip(u, 0.0, 1.0)

            ctrl = np.zeros(self.model.nu)
            for act_id, ui in zip(self.actuator_ids, u):
                ctrl[act_id] = ui

            data.ctrl[:] = ctrl
            mj_step(self.model, data)

            time_log.append(data.time)
            for i in range(num_muscles):
                length_log[i].append(l[i])

            cost = self.cost_fn(data, com_init_height=self.com_init_height)

            total_cost += cost 

            cop = compute_cop(self.model, data)

            cop_log.append(cop)

            if render:
                renderer.update_scene(data)
                frames.append(renderer.render())

        if render:
            video_path = os.path.join(self.video_dir, f"{self.model_name}_optimized.mp4")
            skvideo.io.vwrite(video_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
            renderer.close()

        if plot:
            fig, axes = plt.subplots(num_muscles, 1, figsize=(8, 2*num_muscles), sharex=True)
            if num_muscles == 1:
                axes = [axes]
            time_log = np.array(time_log)
            for i, muscle in enumerate(self.muscles):
                axes[i].plot(time_log, length_log[i], label="Actual length")
                axes[i].plot(time_log, [self.target_len[i]]*len(time_log), 'r--', label="Target length")
                axes[i].set_ylabel(f"{muscle}\n[m]")
                axes[i].legend()
                axes[i].grid(True)
            axes[-1].set_xlabel("Time [s]")
            plt.tight_layout()
            plot_path = os.path.join(self.plot_dir, f"{self.model_name}.png")
            plt.savefig(plot_path)
            plt.close(fig)

        total_cost += cop_cost(cop_log)

        return total_cost

    def optimize(self, x0=None, sigma0=2, maxiter=50, popsize=40, delay_time=0.01, noise_std=0.001):
        num_muscles = len(self.muscles)
        n_params = 3 * num_muscles # Kp. Kd, target_len

        if x0 is None:
            x0 = [1]*num_muscles + [1]*num_muscles + self.init_len

        bounds_lower = [0]*num_muscles + [0]*num_muscles + [0.0]*num_muscles
        bounds_upper = [30]*num_muscles + [30]*num_muscles + [0.6]*num_muscles

        opts = {
            "maxiter": maxiter,
            "popsize": popsize,
            "tolfun": 1e-12,
            "tolx": 1e-12,
            "bounds": [bounds_lower, bounds_upper]  # 0 ≤ Kp,Kd ≤ 30, 0.0 ≤ target_len ≤ 0.6
        }

        es = CMAEvolutionStrategy(x0, sigma0, opts)

        while not es.stop():
            solutions = es.ask()
            costs = [self.run_simulation(s, delay_time=delay_time, noise_std=noise_std) for s in solutions]
            es.tell(solutions, costs)
            es.disp()

        self.best_params = es.result.xbest

        print("done")
        return self.best_params


def compute_cop(model, data):
    """Compute Center of Pressure (COP) from ground reaction forces"""
    total_fz = 0.0
    cop_x, cop_y = 0.0, 0.0

    for i in range(data.ncon):
        con = data.contact[i]
        # geom の名前を取得
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)

        # ground との接触のみ対象
        if "ground" not in (name1 or "") and "ground" not in (name2 or ""):
            continue

        # 接触点の位置（world座標）
        pos = con.pos

        # 接触力（world座標に変換する）
        force = np.zeros(6)  # [fx, fy, fz, mx, my, mz]
        mujoco.mj_contactForce(model, data, i, force)
        f = force[:3]

        # 垂直成分が支持力
        fz = f[2]
        if fz > 1e-6:  # 接触しているときだけ
            total_fz += fz
            cop_x += pos[0] * fz
            cop_y += pos[1] * fz

    if total_fz > 1e-6:
        return np.array([cop_x / total_fz, cop_y / total_fz])
    else:
        return np.zeros(2)  # 接触なし


def cop_cost(cop_log):
    
    cop_cost = 0.0

    if len(cop_log) > 1:
        cop_log = np.array(cop_log)
        diffs = np.diff(cop_log, axis=0)
        cop_cost = np.sum(np.linalg.norm(diffs, axis=1))

    return cop_cost


# def convex_hull_polygon_xy(points_xy):
#     if len(points_xy) < 3:
#         return None
#     pts = np.array(points_xy)
#     hull = ConvexHull(pts)
#     return Polygon(pts[hull.vertices])

# def compute_bos_polygon(model, data):
#     """接触点から BOS を計算（接触がなければ None を返す）"""
#     contact_xy = []
#     for i in range(data.ncon):
#         con = data.contact[i]
#         # ground-plane と足裏 geom の接触だけを取る
#         name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
#         name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)
#         if "ground" in (name1 or "") or "ground" in (name2 or ""):
#             p = con.pos  # world 3D
#             contact_xy.append((p[0], p[1]))

#     if len(contact_xy) < 3:
#         return None  # BOS 不明

#     return convex_hull_polygon_xy(contact_xy)

