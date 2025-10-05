import mujoco
import myosuite
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
from myosuite.utils import gym
import skvideo.io
import os

model = MjModel.from_xml_path("myo_sim/arm26/arm26_cvt3.xml")
data = MjData(model)

renderer = Renderer(model, height=400, width=400)
frames = []

# 肘の角度と角速度のID
elbow_qpos_id = model.jnt_qposadr[model.joint("r_elbow_flex").id]
elbow_qvel_id = model.jnt_dofadr[model.joint("r_elbow_flex").id]

# 目標角度とPDゲイン
target_angle = 1.5
Kp = 1.5
Kd = 0.05

muscles = ["BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed"]
tendon_ids = [model.tendon(f"{m}_tendon").id for m in muscles]

for i in range(model.nu):
    print(f"{i}: {model.actuator(i).name}")

for t in range(300):
    # 現在の肘の角度と速度
    qpos_elbow = data.qpos[elbow_qpos_id]
    qvel_elbow = data.qvel[elbow_qvel_id]

    # 目標角度との誤差とその微分（角速度）
    error = target_angle - qpos_elbow
    derror = -qvel_elbow

    # PD制御信号
    u = Kp * error + Kd * derror
    u = np.clip(u, -1.0, 1.0)

    ctrl = np.zeros(model.nu)

    if u >= 0:
        # 正の時、肘を曲げる（屈曲）→ 二頭筋群とBRAに入力
        ctrl[model.actuator("BIClong").id] = u
        ctrl[model.actuator("BICshort").id] = u
        ctrl[model.actuator("BRA").id] = u
    else:
        ctrl[model.actuator("TRIlong").id] = -u
        ctrl[model.actuator("TRIlat").id] = -u
        ctrl[model.actuator("TRImed").id] = -u


    data.ctrl[:] = ctrl

    print(data.ten_length[tendon_ids])

    # シミュレーションステップ
    mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)


os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/arm26.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})