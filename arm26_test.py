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

muscles = ["BIClong", "BICshort", "BRA", "TRIlong", "TRIlat", "TRImed"]

tendon_ids = [model.tendon(f"{m}_tendon").id for m in muscles]
actuator_ids = [model.actuator(m).id for m in muscles]

target_len = [0.37728848, 0.29647317, 0.12498923, 0.30984061, 0.19891575, 0.18654356]

Kp = [10, 10, 10, 10, 10, 10]
Kd = [1, 1, 1, 1, 1, 1]

for i in range(model.nu):
    print(f"{i}: {model.actuator(i).name}")

for t in range(300):
    # 筋長と速度
    l = data.ten_length[tendon_ids]
    v = data.ten_velocity[tendon_ids]  # 筋長変化速度

    diff_len = l - target_len
    
    # PD制御（筋活性度-1, 1にクリップ）
    u = [(kp * dl) + (kd * vv) for kp, dl, kd, vv in zip(Kp, diff_len, Kd, v)]

    u = np.clip(u, -1.0, 1.0)

    print(u)

    ctrl = np.zeros(model.nu)

    for act_id, ui in zip(actuator_ids, u):
        if ui > 0:
            ctrl[act_id] = ui

    data.ctrl[:] = ctrl

    # シミュレーションステップ
    mj_step(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/arm26_test.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})