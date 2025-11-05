import os
import numpy as np
import mujoco
import skvideo.io
from mujoco import MjModel, MjData, mj_step, Renderer, mj_forward
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from functools import partial

# ======= ユーザ提供データ（省略せずそのまま） =======
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

# ======= ユーティリティ（インデックスを事前計算して高速化） =======
def compress_params(full_params, muscles, pairs):
    num_muscles = len(muscles)
    Kp = full_params[:num_muscles]
    Kd = full_params[num_muscles:2*num_muscles]
    target_len = full_params[2*num_muscles:]

    indices_r = [muscles.index(r) for r, _ in pairs]
    return np.concatenate([Kp[indices_r], Kd[indices_r], target_len[indices_r]])


def expand_params_from_indices(compressed_params, muscles_len, pair_r_indices, pair_l_indices):
    """ 事前に計算した r/l インデックスを使って展開（高速） """
    num_pairs = len(pair_r_indices)
    Kp_r = compressed_params[:num_pairs]
    Kd_r = compressed_params[num_pairs:2*num_pairs]
    target_len_r = compressed_params[2*num_pairs:]

    Kp = np.zeros(muscles_len, dtype=float)
    Kd = np.zeros(muscles_len, dtype=float)
    target_len = np.zeros(muscles_len, dtype=float)

    # vectorized assignment using numpy fancy indexing
    r_idx = np.array(pair_r_indices, dtype=int)
    l_idx = np.array(pair_l_indices, dtype=int)

    Kp[r_idx] = Kp_r
    Kp[l_idx] = Kp_r
    Kd[r_idx] = Kd_r
    Kd[l_idx] = Kd_r
    target_len[r_idx] = target_len_r
    target_len[l_idx] = target_len_r

    return np.concatenate([Kp, Kd, target_len])


# ======= 並列ワーカのグローバル状態（各プロセスで一度だけ初期化） =======
_WORKER = {}

def _init_worker(model_path, muscles, pairs, sim_steps, model_key_qpos):
    """Pool の initializer で一度だけ呼ぶ。各ワーカーでモデルをロードして必要インデックスを保持"""
    model = MjModel.from_xml_path(model_path)

    tendon_ids = np.array([model.tendon(f"{m}_tendon").id for m in muscles], dtype=int)
    actuator_ids = np.array([model.actuator(m).id for m in muscles], dtype=int)

    # precompute pair indices for expansion
    pair_r_indices = [muscles.index(r) for r, l in pairs]
    pair_l_indices = [muscles.index(l) for r, l in pairs]

    # store frequently used constants in global var for worker
    _WORKER['model'] = model
    _WORKER['tendon_ids'] = tendon_ids
    _WORKER['actuator_ids'] = actuator_ids
    _WORKER['pair_r_indices'] = pair_r_indices
    _WORKER['pair_l_indices'] = pair_l_indices
    _WORKER['muscles_len'] = len(muscles)
    _WORKER['sim_steps'] = sim_steps
    _WORKER['model_qpos0'] = model_key_qpos.copy()
    # preallocate renderer = None; will create lazily if asked to render
    _WORKER['renderer'] = None

# ======= ベクトル化された代謝エネルギー計算（numpy） =======
def muscle_metabolic_energy_vectorized(m, A, length_opt_arr, V_CE, F_CE, r=0.5):
    """
    全て numpy 配列で計算する高速版
    """
    m = np.asarray(m)
    A = np.asarray(A)
    li = np.asarray(length_opt_arr)
    vi = np.asarray(V_CE)
    fi = np.asarray(F_CE)

    # avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        v_norm = np.where(li != 0, vi / li, 0.0)

    AMdot = 89.0 * m * (A ** 0.6)

    # Sdot: piecewise depending on v_norm >= 0 (shortening) or < 0 (lengthening)
    pos_mask = v_norm >= 0
    # when shortening (v_norm >= 0)
    alphaS_fast = 15.3
    alphaS_slow = 25.0
    alphaL = 100.0

    Sdot = np.empty_like(v_norm)
    Sdot[pos_mask] = np.abs(m[pos_mask] * ( -((alphaS_slow * v_norm[pos_mask] * r) +
                                              (alphaS_fast * v_norm[pos_mask] * (1 - r))) * A[pos_mask]**2))
    Sdot[~pos_mask] = np.abs(m[~pos_mask] * (alphaL * (-v_norm[~pos_mask]) * A[~pos_mask]))

    Wdot = np.abs(fi * vi)

    Edot = np.sum(AMdot + Sdot + Wdot)
    return Edot


def com_cost(com_log):
    com_log = np.array(com_log)
    diffs = np.diff(com_log, axis=0)
    com_cost = np.sum(np.linalg.norm(diffs, axis=1))
    return com_cost

# ======= シミュレーション実行（ワーカー側で呼ばれる） =======
def run_simulation_worker(compressed_params, delay_time=0.0, noise_std=0.0, camera=None, video_dir=None, model_name=None):
    """
    この関数は各ワーカー内で実行される（_WORKER を参照）
    compressed_params: 圧縮（右側のみ）パラメータ
    """
    model = _WORKER['model']
    tendon_ids = _WORKER['tendon_ids']
    actuator_ids = _WORKER['actuator_ids']
    sim_steps = _WORKER['sim_steps']
    muscles_len = _WORKER['muscles_len']
    pair_r_indices = _WORKER['pair_r_indices']
    pair_l_indices = _WORKER['pair_l_indices']
    qpos0 = _WORKER['model_qpos0']

    # 展開（高速版）
    params_full = expand_params_from_indices(compressed_params, muscles_len, pair_r_indices, pair_l_indices)
    num_muscles = muscles_len

    Kp = params_full[:num_muscles]
    Kd = params_full[num_muscles:2*num_muscles]
    target_len = params_full[2*num_muscles:]

    data = MjData(model)
    data.qpos[:] = qpos0
    data.qvel[:] = 0
    data.qacc[:] = 0
    mj_forward(model, data)

    total_Edot = 0.0
    dt = model.opt.timestep
    delay_steps = max(1, int(delay_time / dt))

    length_buffer = []
    velocity_buffer = []

    baseline = 0.05
    reflex_gain = 2.0

    frames = []
    renderer = None

    com_log = []

    for step in range(sim_steps):
        l = data.ten_length[tendon_ids]
        v = data.ten_velocity[tendon_ids]
        f = data.actuator_force[tendon_ids]

        # push buffers (avoid copies where possible)
        length_buffer.append(l.copy())
        velocity_buffer.append(v.copy())

        if len(length_buffer) > delay_steps:
            l_delayed = length_buffer[-delay_steps]
            v_delayed = velocity_buffer[-delay_steps]
        else:
            l_delayed = l
            v_delayed = v

        diff_len = l_delayed - target_len
        # vectorized control computation
        reflex = reflex_gain * np.clip(diff_len, 0, None)
        u = baseline + Kp * diff_len + Kd * v + reflex

        if noise_std > 0:
            u = u + np.random.normal(0.0, noise_std, size=u.shape)

        np.clip(u, 0.0, 1.0, out=u)

        # set ctrl vector using precomputed actuator ids
        ctrl = np.zeros(model.nu, dtype=float)
        ctrl[actuator_ids] = u
        data.ctrl[:] = ctrl

        mj_step(model, data)

        if step > 5:
            com = np.array(data.subtree_com[0])
            com_log.append(com)

        total_Edot += muscle_metabolic_energy_vectorized(
            muscle_mass[tendon_ids], u, length_opt[tendon_ids], v, f
        )

        # rendering (必要な時のみ)
        if camera is not None:
            if renderer is None:
                # lazy renderer を使う（各ワーカーで1つだけ）
                renderer = Renderer(model, height=400, width=400)
            # update tendon colors (これは少しコストが高いので最低限に)
            for idx, ui in zip(tendon_ids, u):
                r = float(ui); g = 0.0; b = float(1.0 - ui); a = 1.0
                model.tendon_rgba[idx] = np.array([r, g, b, a])
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())

    # video 書き出し（呼ばれたら）
    if camera is not None and len(frames) > 0 and video_dir is not None:
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, f"{model_name}_{camera}.mp4")
        skvideo.io.vwrite(video_path, np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})

    # simple COM cost: ここは軽量化のために単純版（必要なら元の重い版に戻せます）
    # NOTE: model.subtree_com から realtime にアクセスするのは高コストなので省略可能
    # ここでは代わりに 0 を返すようにしているが、もし COM を取りたいなら mj_forward 後に data.subtree_com を読める
    total_cost = np.log1p(np.exp((1e2 * com_cost(com_log)) - 4)) + (1e-5 * total_Edot)

    return float(total_cost)

# ======= 最適化管理クラス =======
class PDGainOptimizer:
    def __init__(self, model_path, muscles, sim_steps=100,
                 model_name=None, output_dir="results", cost_fn=None):
        self.model_path = model_path
        self.model = MjModel.from_xml_path(model_path)  # メインプロセスでも一度ロードして情報取得
        self.muscles = muscles
        self.tendon_ids = [self.model.tendon(f"{m}_tendon").id for m in muscles]
        self.actuator_ids = [self.model.actuator(m).id for m in muscles]

        data = MjData(self.model)
        data.qpos[:] = self.model.key_qpos[0].copy()
        data.qvel[:] = 0
        data.qacc[:] = 0
        mj_forward(self.model, data)
        self.com_init_height = float(np.array(data.subtree_com[0])[2])

        self.init_len = np.array([data.ten_length[t_id] for t_id in self.tendon_ids])
        print("Initial tendon lengths:", self.init_len)

        self.sim_steps = sim_steps
        self.model_name = model_name if model_name else os.path.splitext(os.path.basename(model_path))[0]
        self.output_dir = output_dir

        if cost_fn is None:
            self.cost_fn = lambda *args, **kwargs: 0.0
        else:
            self.cost_fn = cost_fn

        self.video_dir = os.path.join(output_dir, "videos")
        os.makedirs(self.video_dir, exist_ok=True)

    def optimize(self, x0=None, sigma0=2, maxiter=50, popsize=40,
                 delay_time=0.0, noise_std=0.0, n_jobs=None, camera=None):
        num_pairs = len(pairs)

        self.cost_history = []

        if x0 is None:
            init_Kp = [1.0]*num_pairs
            init_Kd = [1.0]*num_pairs
            init_target = self.init_len[[self.muscles.index(r) for r, l in pairs]]
            x0 = np.concatenate([init_Kp, init_Kd, init_target])

        bounds_lower = [0]*num_pairs + [0]*num_pairs + [0.0]*num_pairs
        bounds_upper = [15]*num_pairs + [15]*num_pairs + [0.6]*num_pairs

        opts = {
            "maxiter": maxiter,
            "popsize": popsize,
            "tolfun": 1e-12,
            "tolx": 1e-12,
            "bounds": [bounds_lower, bounds_upper]
        }

        es = CMAEvolutionStrategy(x0, sigma0, opts)

        if n_jobs is None:
            n_jobs = max(1, mp.cpu_count() // 2)

        print(f"🔧 Using {n_jobs} parallel workers")

        # precompute pair indices to pass to workers
        pair_r_indices = [self.muscles.index(r) for r, l in pairs]
        pair_l_indices = [self.muscles.index(l) for r, l in pairs]

        # ワーカー初期化を partial にして model path 等を渡す
        initargs = (self.model_path, self.muscles, pairs, self.sim_steps, self.model.key_qpos[0])

        # spawn でプロセスを立ち上げ（Windows や互換性を考慮）
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_jobs, initializer=_init_worker, initargs=initargs) as pool:
            # prepare worker function with fixed kwargs
            worker_func = partial(run_simulation_worker, delay_time=delay_time, noise_std=noise_std,
                                  camera=camera, video_dir=self.video_dir, model_name=self.model_name)

            while not es.stop():
                solutions = es.ask()
                # solutions は圧縮パラメータ（右側のみ）の配列
                # 並列実行：map を使って複数のワーカへ分配（apply_async のオーバーヘッドを減らす）
                costs = pool.map(worker_func, solutions)
                es.tell(solutions, costs)
                es.disp()

                mean_cost = float (np.mean(costs))
                self.cost_history.append(mean_cost)

        best = es.result.xbest

        # 最適化後、動画生成の前に初期化
        _init_worker(self.model_path, self.muscles, pairs, self.sim_steps, self.model.key_qpos[0])

        # 動画生成
        for cam in ["front", "diagonal", "side", "oblique"]:
            run_simulation_worker(best, delay_time=delay_time, noise_std=noise_std, camera=cam, video_dir=self.video_dir, model_name=self.model_name)

        # 最後にフル展開（メインプロセス用）して保持
        self.best_params = expand_params_from_indices(best, len(self.muscles), pair_r_indices, pair_l_indices)
        return self.best_params
