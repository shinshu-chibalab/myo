import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
import skvideo.io
import os
import multiprocessing as mp

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

# モデル読み込み
model = MjModel.from_xml_path("myo_sim/gait2354/gait2354_simbody_cvt3.xml")
data = MjData(model)

# キーフレーム初期化
data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0
data.qacc[:] = 0

renderer = Renderer(model, height=400, width=400)
frames = []

# --- 初期状態を反映 ---
mujoco.mj_forward(model, data)

# 各筋のアクチュエータIDを一括取得
actuator_ids = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r)
    for r, l in pairs
], dtype=int)

# 対応する lengthrange を NumPy スライスでまとめて取得
length_range = model.actuator_lengthrange[actuator_ids]

print(length_range)

# print(mp.cpu_count())
# print(float(np.array(data.subtree_com[0][2])))
# for i in range(model.ngeom):
#     print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))

for t in range(50):
    ctrl = np.zeros(model.nu)
    data.ctrl[:] = ctrl

    # 接触している geom の確認
    contacts = []

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        contacts.append((name1, name2))

        pos = con.pos

        x = pos[0]
        y = pos[1]

        print(f"step={t}, name1={name1}, name2={name2}, [x, y]={x, y}")

    mj_step(model, data)
    renderer.update_scene(data)
    frames.append(renderer.render())

os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/kakunin.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})

x0 = [10] * 5 + [1] * 1
# print(x0)
