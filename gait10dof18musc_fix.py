import xml.etree.ElementTree as ET

xml_path = "myo_sim/gait10dof18musc/gait10dof18musc_cvt3.xml"
output_path = "myo_sim/gait10dof18musc/gait10dof18musc_fixed.xml"

fixed_joints = [
    "pelvis_tx", "pelvis_ty", "pelvis_tilt",
    "hip_flexion_r", "knee_r_translation1", "knee_r_translation2", "knee_angle_r",
    "hip_flexion_l", "knee_l_translation1", "knee_l_translation2", "knee_angle_l",
    "lumbar_extension"
]

tree = ET.parse(xml_path)
root = tree.getroot()

for joint in root.findall(".//joint"):
    name = joint.attrib.get("name", "")
    if name in fixed_joints:
        joint.set("limited", "true")
        joint.set("range", "0 1e-6")  # 完全固定ではなく極小範囲

tree.write(output_path)
print(f"固定済みXMLを保存しました: {output_path}")
