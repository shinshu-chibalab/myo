import os
import numpy as np
import mujoco
import matplotlib.pyplot as plt
import skvideo.io
from mujoco import MjModel, MjData, mj_step, Renderer, mj_forward
from cma import CMAEvolutionStrategy
from shapely.geometry import Point, Polygon
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

        if cost_fn is None:
            self.cost_fn = lambda data, diff_len, v, u, step: np.sum(diff_len**2)
        else:
            self.cost_fn = cost_fn

        self.video_dir = os.path.join(output_dir, "videos")
        self.plot_dir = os.path.join(output_dir, "plots")
        os.makedirs(self.video_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        self.bos_active = False

    def run_simulation(self, params, render=False, plot=False,
                       delay_time=0.01, noise_std=0.001):
        data = MjData(self.model)
        num_muscles = len(self.muscles)

        Kp = np.array(params[:num_muscles])
        Kd = np.array(params[num_muscles:2*num_muscles])
        target_len = np.array(params[2*num_muscles:])
        total_cost = 0.0

        data.qpos[:] = self.model.key_qpos[0]
        data.qvel[:] = 0
        data.qacc[:] = 0

        dt = self.model.opt.timestep
        delay_steps = max(1, int(delay_time / dt))

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

            cop = compute_cop(self.model, data)
            if cop is not None:
                cop_log.append(cop)

            if render:
                renderer.update_scene(data)
                frames.append(renderer.render())

        if render:
            video_path = os.path.join(self.video_dir, f"{self.model_name}_optimized.mp4")
            skvideo.io.vwrite(video_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
            renderer.close()

        total_cost += cop_cost(cop_log)
        return total_cost

    def optimize(self, x0=None, sigma0=2, maxiter=50, popsize=40,
                 delay_time=0.01, noise_std=0.001):
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

        while not es.stop():
            solutions = es.ask()
            costs = [self.run_simulation(expand_params(s, self.muscles, pairs),
                                         delay_time=delay_time,
                                         noise_std=noise_std) for s in solutions]
            es.tell(solutions, costs)
            es.disp()

        best = es.result.xbest
        self.best_params = expand_params(best, self.muscles, pairs)

        print("done")
        return self.best_params


def compute_cop(model, data):
    """Compute Center of Pressure (COP) from ground reaction forces"""
    total_fz = 0.0
    cop_x, cop_y = 0.0, 0.0

    for i in range(data.ncon):
        con = data.contact[i]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2)

        if "ground" not in (name1 or "") and "ground" not in (name2 or ""):
            continue

        pos = con.pos
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        f = force[:3]

        fz = f[2]
        if fz > 1e-6:
            total_fz += fz
            cop_x += pos[0] * fz
            cop_y += pos[1] * fz

    if total_fz > 1e-6:
        return np.array([cop_x / total_fz, cop_y / total_fz])
    else:
        return None


def cop_cost(cop_log):
    cop_cost = 0.0
    if len(cop_log) > 1:
        cop_log = np.array(cop_log)
        diffs = np.diff(cop_log, axis=0)
        cop_cost = np.sum(np.linalg.norm(diffs, axis=1))
    return cop_cost / len(cop_log)
