import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step
import skvideo.io
import os
import multiprocessing as mp
import matplotlib.pyplot as plt

l_mtu = np.array([
    0.338137, 0.526421, 0.191264, 0.249268, 0.156724, 0.27299, 
    0.202788, 0.28868, 0.383692, 0.508395, 0.173169, 0.25581,
    0.354064, 0.493252, 0.228134, 0.326673, 0.259564, 0.352614
    ])
l_mtu_opt = np.array([0.435, 0.2482473395, 0.274, 0.2590268069, 0.4235661956, 0.2228807695, 0.4435530664, 0.2953153894, 0.3206270804])
l_opt = np.array([0.109, 0.173, 0.147, 0.1, 0.114, 0.087, 0.06, 0.05, 0.098])
alpha = np.array([0, 0.40142572999999998, 0, 0.13962633999999999, 0.087266460000000004, 0.052359879999999998, 0.29670596999999999, 0.43633231, 0.087266460000000004])

# for i in range(len(l_mtu)):
#     l_ce_norm = (1.0 + (l_mtu[i] - l_mtu_opt[i//2]) / (l_opt[i//2] * np.cos(alpha[i//2])))
#     print(l_ce_norm)

# for i in range(len(l_mtu)):
#     l_ce_norm = l_mtu[i] / l_mtu_opt[i//2]
#     print(l_ce_norm)

# pairs = [
#     ("hamstrings_r", "hamstrings_l"),
#     ("bifemsh_r", "bifemsh_l"),
#     ("glut_max_r", "glut_max_l"),
#     ("iliopsoas_r", "iliopsoas_l"),
#     ("rect_fem_r", "rect_fem_l"),
#     ("vasti_r", "vasti_l"),
#     ("gastroc_r", "gastroc_l"),
#     ("soleus_r", "soleus_l"),
#     ("tib_ant_r", "tib_ant_l")
# ]

# muscles =[
#     "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r", "gastroc_r", "soleus_r", "tib_ant_r",
#     "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l", "gastroc_l", "soleus_l", "tib_ant_l",
# ]

# pairs = [
#     ("hamstrings_r", "hamstrings_l"),
#     ("bifemsh_r", "bifemsh_l"),
#     ("glut_max_r", "glut_max_l"),
#     ("iliopsoas_r", "iliopsoas_l"),
#     ("rect_fem_r", "rect_fem_l"),
#     ("vasti_r", "vasti_l"),
#     ("gastroc_r", "gastroc_l"),
#     ("soleus_r", "soleus_l"),
#     ("tib_ant_r", "tib_ant_l"),
#     ("ercspn_r", "ercspn_l"),
#     ("intobl_r", "intobl_l"),
#     ("extobl_r", "extobl_l")
# ]

# muscles =[
#     "hamstrings_r", "bifemsh_r", "glut_max_r", "iliopsoas_r", "rect_fem_r", "vasti_r", "gastroc_r", "soleus_r", "tib_ant_r",
#     "hamstrings_l", "bifemsh_l", "glut_max_l", "iliopsoas_l", "rect_fem_l", "vasti_l", "gastroc_l", "soleus_l", "tib_ant_l",
#     "ercspn_r", "ercspn_l", "intobl_r", "intobl_l", "extobl_r", "extobl_l"
# ]

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
# model = MjModel.from_xml_path("myo_sim/gait10dof18musc/gait10dof18musc_cvt5.xml")
# model = MjModel.from_xml_path("myo_sim/gait10dof24musc/gait10dof24musc_cvt1.xml")
model = MjModel.from_xml_path("myo_sim/gait2354/gait2354_simbody_cvt3.xml")
data = MjData(model)

tendon_ids = np.array([model.tendon(f"{m}_tendon").id for m in muscles], dtype=int)
actuator_ids = np.array([model.actuator(m).id for m in muscles], dtype=int)

print(tendon_ids)
print(actuator_ids)

# default-pose を適用
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "default-pose")
mujoco.mj_resetDataKeyframe(model, data, key_id)

renderer = mujoco.Renderer(model, height=400, width=400)
frames = []

data.qpos[:] = model.key_qpos[0].copy()
data.qvel[:] = 0
data.qacc[:] = 0

# --- 初期状態を反映 ---
mujoco.mj_forward(model, data)

l0 = np.array([data.actuator_length[a_id] for a_id in actuator_ids])
print(l0)

def init():
    for gr, gl in [("calcn_r_geom_1","calcn_l_geom_1"),
                ("toes_r_geom_1","toes_l_geom_1")]:
        ir = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gr)
        il = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gl)

        print(f"{gr} z = {data.geom_xpos[ir]}, {gl} z = {data.geom_xpos[il]}, diff = {data.geom_xpos[ir]-data.geom_xpos[il]}")


    for br, bl in [("femur_r", "femur_l"),
                ("tibia_r", "tibia_l"),
                ("foot_r", "foot_l")]:
        ir = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, br)
        il = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, bl)

        print(f"{br} vs {bl}")
        print("  x diff:", data.xpos[ir])
        print("  x diff:", data.xpos[il])

    com = data.subtree_com[0]
    print("COM y =", com[1])

# 各筋のアクチュエータIDを一括取得
actuator_ids = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r)
    for r, l in pairs
], dtype=int)



actuator_ids = [model.actuator(m).id for m in muscles]

actuator_ids_r = np.array([
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, r)
    for r, l in pairs
], dtype=int)

length_range = model.actuator_lengthrange[actuator_ids_r]

print(length_range[:, 0])
print(length_range[:, 1])

# 対応する lengthrange を NumPy スライスでまとめて取得
length_range = model.actuator_lengthrange[actuator_ids]

# print(length_range)

# print(mp.cpu_count())
# print(float(np.array(data.subtree_com[0][2])))
# for i in range(model.ngeom):
#     print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))
com_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "com_marker")
cop_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cop_marker")
torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
print(f"torso_id: {torso_id}")
torso_com_z = data.subtree_com[23][2]

init()

print(data.ten_length[-4])
print(data.ten_length[5])
for i in range(model.nq):
    print(i, model.joint(i).name)

pelvis_tilt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pelvis_tilt")
ankle_angle_r_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_r")
ankle_angle_l_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ankle_angle_l")
lumbar_extension_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "lumbar_extension")

print(pelvis_tilt_id)
print(ankle_angle_r_id)
print(ankle_angle_l_id)

data.qpos[pelvis_tilt_id] -= 0.03
data.qpos[pelvis_tilt_id] -= 0.01
data.qpos[pelvis_tilt_id] -= 0.01

soleus_length_history = []
gastroc_length_history = []

soleus_act_history = []
gastroc_act_history = []

for m in ["ercspn_r", "ercspn_l", "intobl_r", "intobl_l", "extobl_r", "extobl_l"]:
    aid = model.actuator(m).id
    tid = model.tendon(f"{m}_tendon").id
    print(m, "lengthrange=", model.actuator_lengthrange[aid], "ten_length=", data.ten_length[tid])

for t in range(200):
    ctrl = np.zeros(model.nu)
    if t % 10 == 0:
        # ctrl[5] = 0.0
        # ctrl[6] = 0.3
        # ctrl[7] = 0.0
        # ctrl[8] = 0.8
        # ctrl[14] = 0.0
        # ctrl[15] = 0.3
        # ctrl[16] = 0.0
        # ctrl[17] = 0.8
        ctrl[18] = 1.0
        ctrl[19] = 1.0
        ctrl[20] = 1.0
        ctrl[21] = 1.0
        ctrl[22] = 0.2
        ctrl[23] = 0.2
    data.ctrl[:] = ctrl
    soleus_length_history.append(
        data.ten_length[18]
    )

    gastroc_length_history.append(
        data.ten_length[20]
    )

    soleus_act_history.append(
        data.act[7]
    )

    gastroc_act_history.append(
        data.act[6]
    )

    v_t = data.ten_velocity
    l_t = data.ten_length
    f_t = data.actuator_force
    print(f"tendon: l[18]={l_t[18]}, v[18]={v_t[18]}, f[18]={f_t[18]}")
    # print(data.qpos[lumbar_extension_id])
    # COM 計算
    com = data.subtree_com[0].copy()
    if t % 10 == 0:
        torso_com_z = data.subtree_com[23][2]
        print(torso_com_z)

    # ★ geom の位置を更新
    model.geom_pos[com_geom_id] = com

    # 接触している geom の確認
    contacts = []

    total_Fz = 0.0
    weighted_pos = np.zeros(3)

    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
        contacts.append((name1, name2))

        pos = con.pos.copy()

        # 接触力を取り出す（最重要）
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)

        # force = [fx, fy, fz, mx, my, mz]
        fz = force

        # print(f"t={t}, contact: {name1} -- {name2}, F={fz}, pos={pos}")

        if fz[0] > 0:   # 垂直方向に押している接触のみ採用
            weighted_pos += pos * abs(fz[0])
            total_Fz += abs(fz[0])

    if total_Fz > 0:
        cop = weighted_pos / total_Fz
        cop[2] = 0.0
        model.geom_pos[cop_geom_id] = cop
        # print(cop)
        model.geom_rgba[cop_geom_id] = np.array([0, 1, 0, 1])
    else:
        cop = np.array([np.nan, np.nan, np.nan])
        model.geom_rgba[cop_geom_id] = np.array([1, 0, 0, 1])

    mj_step(model, data)
    renderer.update_scene(data, camera="oblique")
    frames.append(renderer.render())

os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/kakunin.mp4", np.asarray(frames), inputdict={"-r": "200"}, outputdict={"-pix_fmt": "yuv420p"})

# time = np.arange(len(soleus_length_history))

# plt.figure(figsize=(10, 5))

# plt.plot(
#     time,
#     soleus_length_history,
#     label="soleus_r length"
# )

# plt.plot(
#     time,
#     gastroc_length_history,
#     label="gastroc_r length"
# )

# plt.xlabel("step")
# plt.ylabel("MTU length [m]")
# plt.title("Muscle-Tendon Length")
# plt.legend()
# plt.grid()

# plt.show()

