"""Compose a task around the unmodified standard YAM robot description."""
import xml.etree.ElementTree as ET
from copy import deepcopy
from .robot_assets import model_xml

# Public, fixed mounting calibration. Each upstream arm is inside a rigid mount.
DUAL_MOUNTS = {"left": ((0, -.32, 0), .65), "right": ((0, .32, 0), -.65)}


def dual_yam_scene(model_name):
    import math
    root = yam_scene(model_name)
    world = root.find("worldbody")
    original = world.find("body[@name='arm']")
    world.remove(original)
    for light in list(world.findall("light")):
        if light.get("target"):
            world.remove(light)
    sections = {name: list(root.find(name)) for name in ("actuator", "equality", "contact")}
    for name in sections:
        root.find(name).clear()
    for side, (position, yaw) in DUAL_MOUNTS.items():
        def renamed(element):
            element = deepcopy(element)
            for node in element.iter():
                for attr in ("name", "joint", "joint1", "joint2", "body1", "body2"):
                    if attr in node.attrib:
                        node.set(attr, side + "_" + node.get(attr))
            return element
        mount = ET.SubElement(world, "body", name=side + "_mount",
                              pos=" ".join(map(str, position)),
                              quat=f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}")
        mount.append(renamed(original))
        for name, elements in sections.items():
            root.find(name).extend(renamed(element) for element in elements)
    return root


def yam_scene(model_name="yam_button_v1"):
    root = ET.fromstring(model_xml())
    root.set("model", model_name)
    # Robot-only keyframe has no button coordinate. No robot dynamics/geometry edits.
    root.remove(root.find("keyframe"))
    root.find("option").set("timestep", "0.002")
    visual = root.find("visual")
    ET.SubElement(visual, "global", offwidth="640", offheight="480")
    ET.SubElement(visual, "quality", shadowsize="2048")
    world = root.find("worldbody")
    for position in ("0.2 -0.5 1.4", "0.6 0.6 1.2"):
        ET.SubElement(world, "light", pos=position, dir="0 0 -1")
    for name, pos, axes in (
        ("front", "1 -1 0.8", "0.8 0.6 0 -0.3 0.4 0.866"),
        ("left", "-0.6 -0.8 0.8", "0.707 -0.707 0 0.408 0.408 0.816"),
        ("overhead", "0.3 0 1.2", "1 0 0 0 1 0"),
    ):
        ET.SubElement(world, "camera", name=name, pos=pos, xyaxes=axes,
                      fovy="58" if name == "overhead" else "50")
    ET.SubElement(world, "geom", name="table", type="box", pos="0.2 0 -0.025",
                  size="0.8 0.6 0.025", rgba="0.65 0.61 0.52 1")
    return root


def button_scene(x: float, y: float) -> str:
    root = yam_scene()
    world = root.find("worldbody")
    button = ET.SubElement(world, "body", name="button_base", pos=f"{x:.9f} {y:.9f} 0")
    ET.SubElement(button, "geom", name="button_housing", type="box", pos="0 0 0.012",
                  size="0.04 0.04 0.012", rgba="0.12 0.12 0.14 1")
    cap = ET.SubElement(button, "body", name="button_cap", pos="0 0 0.037")
    ET.SubElement(cap, "joint", name="button_travel", type="slide", axis="0 0 -1",
                  range="0 0.012", damping="2", stiffness="120", springref="0",
                  armature="0.001", solreflimit="0.004 1", solimplimit="0.999 0.9999 0.0001")
    ET.SubElement(cap, "geom", name="button_cap_geom", type="cylinder", size="0.027 0.009",
                  mass="0.02", rgba="0.9 0.05 0.04 1", friction="0.7 0.01 0.001",
                  solref="0.006 1", solimp="0.95 0.99 0.001")
    ET.SubElement(button, "geom", name="indicator", type="sphere", pos="0.033 0.033 0.029",
                  size="0.004", contype="0", conaffinity="0", rgba="0.2 0.2 0.2 1")
    ET.SubElement(root.find("contact"), "exclude", body1="button_base", body2="button_cap")
    return ET.tostring(root, encoding="unicode")
