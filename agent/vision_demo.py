"""Optional RGB-only diagnostic policy; excluded from the evaluated agent bundle.

Requires Pillow. No trusted simulator imports, scene files, object state, or score
access. Uses only the public robot/camera calibration and permitted API.
"""

import io
import math

from PIL import Image

from robot_client import RobotClient
from robot_kinematics import RobotKinematics


def red_button_xy(png):
    image = Image.open(io.BytesIO(png)).convert("RGB")
    pixels = image.load()
    locations = [(x, y) for y in range(image.height) for x in range(image.width)
                 if pixels[x, y][0] > 100
                 and pixels[x, y][0] > 2 * pixels[x, y][1]
                 and pixels[x, y][0] > 2 * pixels[x, y][2]]
    if len(locations) < 20:
        raise RuntimeError("Cannot identify red button from camera; no motion sent")
    u = sum(x + 0.5 for x, _ in locations) / len(locations)
    v = sum(y + 0.5 for _, y in locations) / len(locations)
    # Table-plane estimate; cap elevation creates a small visual estimation error.
    scale = 2 * 1.2 * math.tan(math.radians(58) / 2) / image.height
    return 0.3 + (u - image.width / 2) * scale, -(v - image.height / 2) * scale


def green_indicator_visible(png):
    image = Image.open(io.BytesIO(png)).convert("RGB")
    return sum(g > 100 and g > 1.8 * r and g > 1.8 * b for r, g, b in image.getdata()) >= 3


def main():
    robot = RobotClient()
    kinematics = RobotKinematics()
    x, y = red_button_xy(robot.observe("overhead"))
    robot.set_gripper(0.012, 1.0)
    trajectory = ((0.12, 3.0), (0.10, 1.0), (0.08, 1.0), (0.06, 1.0),
                  (0.052, 0.5), (0.050, 0.5), (0.048, 0.5),
                  (0.046, 0.5), (0.044, 0.5), (0.042, 0.5))
    for height, duration in trajectory:
        robot.move_joints(kinematics.downward_pose(x, y, height), duration)
        robot.wait(0.2)
        if green_indicator_visible(robot.observe("overhead")):
            break
    print(robot.submit())  # Receipt only: this policy does not receive its score.


if __name__ == "__main__":
    main()
