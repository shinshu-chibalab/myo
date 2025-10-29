import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import skvideo.io
from mujoco import MjModel, MjData, mj_step, Renderer, mj_forward
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from scipy.spatial import ConvexHull


# 左右対称ペア
pairs = [
    ("glut_med1_r", "glut_med1_l"),
    ("glut_med2_r", "glut_med2_l"),
    ("glut_med3_r", "glut_med3_l"),
    ("bifemlh_r", "bifemlh_l"),
    ("bifemsh_r", "bifemsh_l"),
    ("sar_r", "sar_l"),
    ("add_mag2_r", "add_mag2_l"),
    ("tfl_r", "tfl_l"),
    ("pect_r", "pect_l"),
    ("grac_r", "grac_l"),
    ("glut_max1_r", "glut_max1_l"),
    ("glut_max2_r", "glut_max2_l"),
    ("glut_max3_r", "glut_max3_l"),
    ("iliacus_r", "iliacus_l"),
    ("psoas_r", "psoas_l"),
    ("quad_fem_r", "quad_fem_l"),
    ("gem_r", "gem_l"),
    ("peri_r", "peri_l"),
    ("rect_fem_r", "rect_fem_l"),
    ("vas_int_r", "vas_int_l"),
    ("med_gas_r", "med_gas_l"),
    ("soleus_r", "soleus_l"),
    ("tib_post_r", "tib_post_l"),
    ("tib_ant_r", "tib_ant_l"),
    ("ercspn_r", "ercspn_l"),
    ("intobl_r", "intobl_l"),
    ("extobl_r", "extobl_l")
]

muscle_mass = np.array([0.17959950000000002, 0.22130550000000002, 0.19380000000000003, 0.8829, 0.4172759999999999, 0.24336, 0.8505089999999998, 0.066405, 0.0798, 0.171072, 0.244098, 0.36117899999999975, 0.23846399999999998, 0.3219, 0.33390000000000003, 0.06172199999999999, 0.011808, 0.034631999999999996, 0.399798, 1.605, 0.675, 0.6, 0.3348, 0.882, 0.17959950000000002, 0.22130550000000002, 0.19380000000000003, 0.8829, 0.4172759999999999, 0.24336, 0.8505089999999998, 0.066405, 0.0798, 0.171072, 0.244098, 0.36117899999999975, 0.23846399999999998, 0.3219, 0.33390000000000003, 0.06172199999999999, 0.011808, 0.034631999999999996, 0.399798, 1.605, 0.675, 0.6, 0.3348, 0.882, 0.9, 0.9, 0.27, 0.27, 0.324, 0.324])
length_opt = np.array([0.0535, 0.0845, 0.0646, 0.109, 0.173, 0.52, 0.121, 0.095, 0.1, 0.352, 0.142, 0.1469999999999999, 0.144, 0.1, 0.1, 0.054, 0.024, 0.026, 0.114, 0.107, 0.09, 0.05, 0.031, 0.098, 0.0535, 0.0845, 0.0646, 0.109, 0.173, 0.52, 0.121, 0.095, 0.1, 0.352, 0.142, 0.1469999999999999, 0.144, 0.1, 0.1, 0.054, 0.024, 0.026, 0.114, 0.107, 0.09, 0.05, 0.031, 0.098, 0.12, 0.12, 0.1, 0.1, 0.12, 0.12])

def compress_params(full_params, muscles, pairs):
    """左右対称ペアの右側だけ抜き出して圧縮"""
    num_muscles = len(muscles)
    Kp = full_params[:num_muscles]
    Kd = full_params[num_muscles:2*num_muscles]
    target_len = full_params[2*num_muscles:]

    indices_r = [muscles.index(r) for r, l in pairs]
    return np.concatenate([
        Kp[indices_r], Kd[indices_r], target_len[indices_r]
    ])


def expand_params(compressed_params, muscles, pairs):
    """圧縮パラメータを展開して左右両方の配列に戻す"""
    num_pairs = len(pairs)
    Kp_r = compressed_params[:num_pairs]
    Kd_r = compressed_params[num_pairs:2*num_pairs]
    target_len_r = compressed_params[2*num_pairs:]

    Kp = np.zeros(len(muscles))
    Kd = np.zeros(len(muscles))
    target_len = np.zeros(len(muscles))

    for i, (r, l) in enumerate(pairs):
        r_idx = muscles.index(r)
        l_idx = muscles.index(l)
        Kp[r_idx] = Kp[l_idx] = Kp_r[i]
        Kd[r_idx] = Kd[l_idx] = Kd_r[i]
        target_len[r_idx] = target_len[l_idx] = target_len_r[i]

    return np.concatenate([Kp, Kd, target_len])


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
        print("Initial tendon lengths:", self.init_len)

        self.sim_steps = sim_steps
        self.model_name = model_name if model_name else os.path.splitext(os.path.basename(model_path))[0]
        self.output_dir = output_dir
        self.cost_history = []

        if cost_fn is None:
            self.cost_fn = lambda data, diff_len, v, u, step: np.sum(diff_len**2)
        else:
            self.cost_fn = cost_fn

        self.video_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

        self.bos_active = False

    def run_simulation(self, params, camera=None, delay_time=0.0, noise_std=0.0):
        data = MjData(self.model)
        num_muscles = len(self.muscles)

        Kp = np.array(params[:num_muscles])
        Kd = np.array(params[num_muscles:2*num_muscles])
        target_len = np.array(params[2*num_muscles:])
        total_cost = 0.0
        total_Edot = 0.0

        data.qpos[:] = self.model.key_qpos[0]
        data.qvel[:] = 0
        data.qacc[:] = 0

        dt = self.model.opt.timestep
        delay_steps = max(1, int(delay_time / dt))

        length_buffer = []
        velocity_buffer = []

        time_log = []
        length_log = [[] for _ in range(num_muscles)]
        com_log = []

        baseline = 0.05
        reflex_gain = 2.0

        renderer = Renderer(self.model, height=400, width=400) if camera is not None else None
        frames = []

        for step in range(self.sim_steps):
            l = data.ten_length[self.tendon_ids]
            v = data.ten_velocity[self.tendon_ids]
            f = data.actuator_force[self.tendon_ids]

            length_buffer.append(l.copy())
            velocity_buffer.append(v.copy())

            if len(length_buffer) > delay_steps:
                l_delayed = length_buffer[-delay_steps]
                v_delayed = velocity_buffer[-delay_steps]
            else:
                l_delayed = l
                v_delayed = v

            diff_len = l_delayed - target_len
            reflex = reflex_gain * np.clip(diff_len, 0, None)
            u = baseline + Kp * diff_len + Kd * v + reflex

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

            # cost = self.cost_fn(data, com_init_height=self.com_init_height)
            # total_cost += cost

            com = compute_com(data)
            com_log.append(com)

            total_Edot += muscle_metabolic_energy(muscle_mass, u, length_opt, v, f)

            if camera is not None:
                # --- 筋色の更新（u: 0→青, 1→赤） ---
                for t_id, ui in zip(self.tendon_ids, u):
                    # 赤成分を ui に応じて増加、青成分を (1-ui) に応じて増加
                    r = float(ui)
                    g = 0.0
                    b = float(1.0 - ui)
                    a = 1.0

                    # モデルの色を更新
                    self.model.tendon_rgba[t_id] = np.array([r, g, b, a])

                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())

        if camera is not None:
            video_path = os.path.join(self.video_dir, f"{self.model_name}_{camera}.mp4")
            skvideo.io.vwrite(video_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})

        total_cost += (1e4 * com_cost(com_log)) + (total_Edot * 1e-5)
        return total_cost

    def optimize(self, x0=None, sigma0=2, maxiter=50, popsize=40,
                 delay_time=0.0, noise_std=0.0, n_jobs=None):
        num_pairs = len(pairs)

        if x0 is None:
            init_Kp = [1]*num_pairs
            init_Kd = [1]*num_pairs
            init_target = self.init_len[[self.muscles.index(r) for r, l in pairs]]
            x0 = np.concatenate([init_Kp, init_Kd, init_target])

        bounds_lower = [0]*num_pairs + [0]*num_pairs + [0.0]*num_pairs
        bounds_upper = [30]*num_pairs + [30]*num_pairs + [0.6]*num_pairs

        opts = {
            "maxiter": maxiter,
            "popsize": popsize,
            "tolfun": 1e-12,
            "tolx": 1e-12,
            "bounds": [bounds_lower, bounds_upper]
        }

        es = CMAEvolutionStrategy(x0, sigma0, opts)

        # --- 並列処理設定 ---
        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() // 2)

        print(f"🔧 Using {n_jobs} parallel workers")

        # --- プロセスプールを使って並列最適化 ---
        with mp.get_context("spawn").Pool(processes=n_jobs) as pool:
            while not es.stop():
                solutions = es.ask()

                # 並列で run_simulation を実行
                results = [
                    pool.apply_async(
                        self.run_simulation,
                        args=(expand_params(s, self.muscles, pairs),),
                        kwds={'delay_time': delay_time, 'noise_std': noise_std}
                    )
                    for s in solutions
                ]

                # 結果を取得
                costs = [r.get() for r in results]

                self.cost_history.append(np.mean(costs))

                es.tell(solutions, costs)
                es.disp()

        best = es.result.xbest
        self.best_params = expand_params(best, self.muscles, pairs)

        return self.best_params

def compute_com(data):
    # COM(重心)　xyz
    return np.array(data.subtree_com[0])

def com_cost(com_log):
    com_log = np.array(com_log)
    diffs = np.diff(com_log, axis=0)
    com_cost = np.sum(np.linalg.norm(diffs, axis=1))
    return com_cost / len(com_log)



def muscle_metabolic_energy(m, A, length_opt, V_CE, F_CE, r=0.5, S=1.0):
    """
    m: muscle mass [kg]
    A: activation (0-1)
    v_norm: normalized fiber velocity (v_CE / l_CE_opt)
    F_CE_norm: normalized muscle force
    r: slow-twitch ratio (0-1)
    S: aerobic scaling factor
    """

    # Shortening / lengthening heat
    alphaS_fast = 15.3
    alphaS_slow = 25
    alphaL = 100

    Edot = 0

    # ✅ zipで各筋ごとに値を取り出す
    for mi, Ai, li, vi, fi in zip(m, A, length_opt, V_CE, F_CE):

        v_norm = vi / li if li != 0 else 0.0

        # Activation & maintenance heat（簡易版）
        AMdot = mi * ((128 * (1 - r) + 25) * (Ai ** 0.6) * S)

        # Shortening / lengthening heat
        if v_norm >= 0:
            Sdot = mi * ( -((alphaS_slow * v_norm * r) +
                             (alphaS_fast * v_norm * (1 - r))) * Ai**2 * S )
        else:
            Sdot = mi * (alphaL * (-v_norm) * Ai * S)

        # Mechanical work rate（簡易）
        Wdot = abs(fi * vi)

        Edot += (AMdot + Sdot + Wdot)

    return Edot
