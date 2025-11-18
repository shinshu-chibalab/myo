import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
import os
import skvideo.io

model = MjModel.from_xml_path("myo_sim/gait10dof18musc/gait10dof18musc_fixed.xml")  # モデルファイルを指定
data = MjData(model)
renderer = Renderer(model, height=400, width=400)

# keyframe を適用
mujoco.mj_resetDataKeyframe(model, data, 0)
data.qpos[1] += 0.2
mujoco.mj_forward(model, data)


frames = []
T = 300

for t in range(T):
    ctrl = np.zeros(model.nu)

    # フェーズを単純に時間で分ける（前半: 立脚期, 後半: 遊脚期）
    if t < T // 2:
        # 🦶 立脚期（股関節・膝伸展、足底屈）
        ctrl[model.actuator("glut_max_r").id] = 0.6
        ctrl[model.actuator("vasti_r").id] = 0.6
        ctrl[model.actuator("soleus_r").id] = 0.5
    else:
        # 🚶‍♂️ 遊脚期（股関節屈曲、足背屈）
        # ctrl[model.actuator("psoas_r").id] = 0.5
        ctrl[model.actuator("rect_fem_r").id] = 0.5
        ctrl[model.actuator("tib_ant_r").id] = 0.4

    data.ctrl[:] = ctrl
    mj_step(model, data)

    renderer.update_scene(data)
    pixels = renderer.render()
    frames.append(pixels)

# 動画として保存
os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/step_forward_leg6dof9musc.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
