import sys
import numpy as np
import xml.etree.ElementTree as ET

from smpl_viz.joints import BODY_CHAIN

np.set_printoptions(4, threshold=sys.maxsize, suppress=True, linewidth=np.inf)


def _make_body_node(tag, position_str, jtype="ball", add_geom=True):
    node = ET.Element('body', name=tag, pos=position_str)
    if jtype:
        ET.SubElement(node, 'joint', type=jtype)
    if add_geom:
        ET.SubElement(node, 'geom')
    return node


def _attach_children(parent_node, tags, offsets, adjacency, idx=0):
    if idx >= len(tags):
        return
    pos_str = np.array2string(offsets[idx])[1:-1]
    jtype = "free" if idx == 0 else "ball"
    node = _make_body_node(tags[idx], pos_str, jtype=jtype)
    parent_node.append(node)
    for child_idx in np.where(adjacency[0] == idx)[0]:
        _attach_children(node, tags, offsets, adjacency, child_idx)


def _append_ground(xml_root):
    asset = ET.SubElement(xml_root, 'asset')
    ET.SubElement(asset, 'texture', type="skybox", builtin="gradient",
                  rgb1="0.3 0.5 0.7", rgb2="0 0 0", width="512", height="3072")
    ET.SubElement(asset, 'texture', type="2d", name="groundplane", builtin="checker",
                  mark="edge", rgb1="0.2 0.3 0.4", rgb2="0.1 0.2 0.3",
                  markrgb="0.8 0.8 0.8", width="300", height="300")
    ET.SubElement(asset, 'material', name="groundplane", texture="groundplane",
                  texuniform="true", texrepeat="5 5", reflectance="0.2")
    worldbody = ET.SubElement(xml_root, 'worldbody')
    ET.SubElement(worldbody, 'light', pos="0 0 3.5", dir="0 0 -1", directional="true")
    ET.SubElement(worldbody, 'geom', name="floor", size="0 0 0.05", type="plane",
                  material="groundplane", condim='3', conaffinity='15')
    return xml_root


def assemble_mjcf(kintree, joint_positions, skin_weights, face_indices, vert_positions):
    """Build a MuJoCo XML string from SMPL-H mesh data.

    Returns (xml_string, xml_root_element).
    """
    rel_offsets = joint_positions.copy()
    rel_offsets[1:] -= joint_positions[kintree[0, 1:]]

    root = ET.Element('mujoco')

    vis = ET.SubElement(root, 'visual')
    ET.SubElement(vis, 'headlight',
                  attrib={"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(vis, 'rgba', attrib={"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(vis, 'global', attrib={"azimuth": "150", "elevation": "-25"})

    ET.SubElement(root, 'statistic', meansize="0.08")
    ET.SubElement(ET.SubElement(root, 'option'), 'flag', gravity="disable")
    ET.SubElement(ET.SubElement(root, 'default'), 'geom', size="0.01", contype="0")

    wb = ET.SubElement(root, 'worldbody')
    _attach_children(wb, BODY_CHAIN, rel_offsets, kintree)

    deform = ET.SubElement(root, 'deformable')
    flat_verts = np.array2string(vert_positions.reshape(-1))[1:-1]
    flat_faces = np.array2string(face_indices.reshape(-1))[1:-1]
    skin = ET.SubElement(deform, 'skin', rgba="1 1 1 0.5",
                         vertex=flat_verts, face=flat_faces)

    for jid in range(joint_positions.shape[0]):
        w_full = skin_weights[:, jid]
        vids = np.nonzero(w_full)[0]
        ET.SubElement(skin, 'bone',
                      body=BODY_CHAIN[jid],
                      bindpos=np.array2string(joint_positions[jid])[1:-1],
                      bindquat="1 0 0 0",
                      vertid=np.array2string(vids)[1:-1],
                      vertweight=np.array2string(w_full[vids])[1:-1])

    ET.indent(ET.ElementTree(root), space="  ", level=0)
    return ET.tostring(root, encoding='unicode', method='xml'), root
