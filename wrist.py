import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
import skvideo.io
import os

# モデル読み込み
model = MjModel.from_xml_path("myo_sim/gait2354/gait2354_simbody_cvt3.xml")
data = MjData(model)

# キーフレーム初期化
data.qpos[:] = model.key_qpos[0]
data.qvel[:] = 0
data.qacc[:] = 0

renderer = Renderer(model, height=400, width=400)
frames = []

for t in range(200):
    ctrl = np.zeros(model.nu)
    data.ctrl[:] = ctrl

    mj_step(model, data)
    renderer.update_scene(data, camera="front")
    frames.append(renderer.render())

# 動画保存
os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("videos/gait2354_cam.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
