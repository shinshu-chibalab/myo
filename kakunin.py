import mujoco
import numpy as np
from mujoco import MjModel, MjData, mj_step, Renderer
import skvideo.io
import os
import multiprocessing as mp

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

# print(mp.cpu_count())
print(float(np.array(data.subtree_com[0][2])))
# for i in range(model.ngeom):
#     print(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i))

for t in range(50):
    ctrl = np.zeros(model.nu)
    data.ctrl[:] = ctrl

    # 接触している geom の確認
    contacts = []

    # for i in range(data.ncon):
    #     con = data.contact[i]
    #     g1, g2 = con.geom1, con.geom2
    #     name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
    #     name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)
    #     contacts.append((name1, name2))

    #     pos = con.pos

    #     x = pos[0]
    #     y = pos[1]

    #     print(x, y)

    mj_step(model, data)
    renderer.update_scene(data)
    frames.append(renderer.render())

os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("./videos/kakunin.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})

x0 = [10] * 5 + [1] * 1
# print(x0)
