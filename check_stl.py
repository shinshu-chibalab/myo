import trimesh
import numpy as np

# STL 読み込み
# mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/foot.stl")   # 右足
# mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/bofoot.stl")   # 右足
# mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/femur_r.stl")   # 右足
# mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/fibula.stl")   # 右足
# mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/pelvis.stl")   # 右足
# mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/talus.stl")   # 右足
mesh = trimesh.load("./myo_sim/gait10dof18musc/Geometry/tibia_r.stl")   # 右足
mesh_mirror = mesh.copy()

# Y軸反転（左右反転：モデル座標系に合わせて調整）
mesh_mirror.apply_scale([1, 1, -1])

# 左足 STL 読み込み
# mesh_l = trimesh.load("./myo_sim/gait2354/Geometry/foot.stl")
# mesh_l = trimesh.load("./myo_sim/gait10dof18musc/Geometry/l_bofoot.stl")
# mesh_l = trimesh.load("./myo_sim/gait10dof18musc/Geometry/femur_l.stl")
# mesh_l = trimesh.load("./myo_sim/gait10dof18musc/Geometry/l_fibula.stl")
# mesh_l = trimesh.load("./myo_sim/gait10dof18musc/Geometry/l_pelvis.stl")
# mesh_l = trimesh.load("./myo_sim/gait10dof18musc/Geometry/l_talus.stl")
mesh_l = trimesh.load("./myo_sim/gait10dof18musc/Geometry/tibia_l.stl")

# 頂点数チェック
print(len(mesh.vertices), len(mesh_l.vertices))

print(mesh.centroid)
print(mesh_l.centroid)

# 最近傍距離で差分評価
dist = mesh_l.nearest.signed_distance(mesh_mirror.vertices)
print("min abs diff:", np.min(np.abs(dist)))
print("mean abs diff:", np.mean(np.abs(dist)))
print("max abs diff:", np.max(np.abs(dist)))
