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

def compute_cop(model, data):
    """Compute Center of Pressure (COP) from ground reaction forces"""
    total_fz = 0.0
    cop_x, cop_y = 0.0, 0.0

    for i in range(data.ncon):
        con = data.contact[i]

        # 名前を取得（Noneは空文字に）
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom1) or ""
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, con.geom2) or ""

        # ground-plane との接触のみ対象
        if "ground-plane" not in (name1 + name2):
            continue

        # 足のどの部位かを確認
        foot_geom = name1 if "ground" not in name1 else name2

        # 接触位置と力
        pos = con.pos
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        f = force[:3]
        fz = f[2]

        if fz > 1e-6:
            total_fz += fz
            cop_x += pos[0] * fz
            cop_y += pos[1] * fz

            print(f"接触: {foot_geom:>15s} | fz={fz:7.4f} | pos=({pos[0]:6.3f}, {pos[1]:6.3f}, {pos[2]:6.3f})")

    # COP計算
    if total_fz > 1e-6:
        cop = np.array([cop_x / total_fz, cop_y / total_fz])
        print(f"→ COP = ({cop[0]:.4f}, {cop[1]:.4f})\n")
        return cop
    else:
        print("⚠ 接触なし\n")
        return None



for t in range(50):
    ctrl = np.zeros(model.nu)
    data.ctrl[:] = ctrl

    cop = compute_cop(model, data)
    
    mj_step(model, data)
    renderer.update_scene(data, camera="front")
    frames.append(renderer.render())

# 動画保存
os.makedirs("videos", exist_ok=True)
skvideo.io.vwrite("videos/gait2354_cam.mp4", np.asarray(frames), outputdict={"-pix_fmt": "yuv420p"})
