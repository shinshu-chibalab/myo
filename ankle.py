import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
import os, skvideo.io

# モデル読み込み
model_path = "myo_sim/gait10dof18musc/gait10dof18musc_fixed.xml"
model = MjModel.from_xml_path(model_path)
data = MjData(model)

data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0
data.qacc[:] = 0

# 足首以外の自由度を固定する関節名リスト
fixed_joint_names = [
    "pelvis_tx", "pelvis_ty", "pelvis_tilt",
    "hip_flexion_r", "hip_flexion_l",
    "knee_r_translation1", "knee_r_translation2",
    "knee_angle_r",
    "knee_l_translation1", "knee_l_translation2",
    "knee_angle_l",
    "lumbar_extension"
]

def fix_joints(data, model, fixed_joint_names):
    """指定関節の角度と速度を毎ステップ固定"""
    for jname in fixed_joint_names:
        j_id = model.joint(jname).id
        qposadr = model.jnt_qposadr[j_id]
        dofadr = model.jnt_dofadr[j_id]
        nq = 1 if model.jnt_type[j_id] == mujoco.mjtJoint.mjJNT_HINGE else 3
        data.qpos[qposadr:qposadr+nq] = 0.0
        data.qvel[dofadr:dofadr+nq] = 0.0

# 筋肉リスト（足首用: soleus, tib_ant など）
muscles = [
    "soleus_r", "tib_ant_r",
    "soleus_l", "tib_ant_l"
]
tendon_ids = [model.tendon(f"{m}_tendon").id for m in muscles]
actuator_ids = [model.actuator(m).id for m in muscles]

# 初期筋長を目標に設定
mj_step(model, data)
target_len = data.ten_length[tendon_ids].copy()

# PDゲイン（簡易）
Kp = np.array([0.05, 0.05, 0.05, 0.05])
Kd = np.array([0.02, 0.02, 0.02, 0.02])

renderer = Renderer(model, height=400, width=400)
frames = []

for step in range(300):
    # 足首以外の関節を固定
    # fix_joints(data, model, fixed_joint_names)

    # 筋長と速度
    l = data.ten_length[tendon_ids]
    v = data.ten_velocity[tendon_ids]
    diff_len = l - target_len

    # PD制御
    u = Kp * diff_len + Kd * v
    u = np.clip(u, -1.0, 1.0)

    ctrl = np.zeros(model.nu)
    for act_id, ui in zip(actuator_ids, u):
        if ui > 0:  # 正側だけ使用（片側筋駆動）
            ctrl[act_id] = ui

    data.ctrl[:] = ctrl
    mj_step(model, data)

    renderer.update_scene(data)
    frames.append(renderer.render())

renderer.close()

# 動画保存
os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/stand_ankle_only.mp4",
                  np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
print("動画を保存しました: videos/stand_ankle_only.mp4")
