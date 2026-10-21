"""Private contact-graph scoring for a released, upright block tower."""
import xml.etree.ElementTree as ET

import numpy as np

from .base import Task, body_is_slow, box_corners, robot_body_ids


class TowerTask(Task):
    task_id = "tower"
    instruction = (
        "Build the tallest freestanding tower using all four colored blocks. "
        "Keep each block's white marked top facing upward. Stack blocks with at least "
        "80% of each upper block's footprint supported by the block below. "
        "Release every block and withdraw the robot; the tower must remain still on the table."
    )

    def build(self, world, root, rng):
        self.half_sizes = [np.array([s, s, h]) for s, h in
                           [(.031, .0225), (.029, .025), (.027, .0275), (.025, .030)]]
        colors = [".9 .23 .15 1", ".2 .55 .85 1", ".95 .7 .15 1", ".3 .7 .3 1"]
        # Stratified rows prevent initial overlap while retaining randomized layout.
        for index, (size, color) in enumerate(zip(self.half_sizes, colors)):
            x = rng.uniform(.27, .36) if index < 2 else rng.uniform(.43, .49)
            y = rng.uniform(-.19, -.10) if index % 2 == 0 else rng.uniform(.10, .19)
            body = ET.SubElement(world, "body", name=f"tower_block_{index}",
                                 pos=f"{x} {y} {size[2] + .001}",
                                 euler=f"0 0 {rng.uniform(-.4,.4)}")
            ET.SubElement(body, "freejoint", name=f"tower_free_{index}")
            ET.SubElement(body, "geom", name=f"tower_geom_{index}", type="box",
                          size=" ".join(map(str, size)), mass=".07", rgba=color,
                          friction="1 .005 .0001", solref=".006 1",
                          solimp=".95 .99 .001")
            ET.SubElement(body, "geom", type="box", size=".009 .009 .00005",
                          pos=f"0 0 {size[2] + .00005}", rgba="1 1 1 1",
                          contype="0", conaffinity="0", mass="0")
        self.maximum_height = float(sum(2 * size[2] for size in self.half_sizes))

    def bind(self, model, data):
        super().bind(model, data)
        self.body_ids = [model.body(f"tower_block_{i}").id for i in range(4)]
        self.geom_ids = [model.geom(f"tower_geom_{i}").id for i in range(4)]
        self.table_geom = model.geom("table").id
        self.robot_ids = robot_body_ids(model)

    @staticmethod
    def _clip(polygon, center, rotation, size):
        # Exact convex polygon/rectangle intersection, including arbitrary yaw.
        output = [(p - center) @ rotation for p in polygon]
        for axis in (0, 1):
            for sign in (-1, 1):
                source, output = output, []
                if not source:
                    break
                previous = source[-1]
                pd = size[axis] - sign * previous[axis]
                for current in source:
                    cd = size[axis] - sign * current[axis]
                    if (cd >= 0) != (pd >= 0):
                        output.append(previous + pd / (pd - cd) * (current - previous))
                    if cd >= 0:
                        output.append(current)
                    previous, pd = current, cd
        if len(output) < 3:
            return 0.0
        points = np.asarray(output)
        return abs(float(np.sum(points[:, 0] * np.roll(points[:, 1], -1) -
                                points[:, 1] * np.roll(points[:, 0], -1)))) / 2

    def evaluate(self, simulation, answer=None):
        positions = [self.data.geom_xpos[g].copy() for g in self.geom_ids]
        rotations = [self.data.geom_xmat[g].reshape(3, 3) for g in self.geom_ids]
        corners = [box_corners(self.model, self.data, g) for g in self.geom_ids]
        bottoms = [float(p[:, 2].min()) for p in corners]
        tops = [float(p[:, 2].max()) for p in corners]
        valid = [bool(r[2, 2] > .999 and abs(bottoms[i]) < 1.0 and
                      bottoms[i] >= -.002 and body_is_slow(self.model, self.data, self.body_ids[i]))
                 for i, r in enumerate(rotations)]
        geom_index = {g: i for i, g in enumerate(self.geom_ids)}
        contacts, table, adjacency, foreign = set(), set(), [set() for _ in range(4)], set()
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            ia, ib = geom_index.get(a), geom_index.get(b)
            if ia is not None and ib is not None:
                adjacency[ia].add(ib)
                adjacency[ib].add(ia)
                contacts.add(tuple(sorted((ia, ib))))
            elif ia is not None or ib is not None:
                i, other = (ia, b) if ia is not None else (ib, a)
                if other == self.table_geom:
                    if -.002 <= bottoms[i] <= .002:
                        table.add(i)
                else:
                    foreign.add(i)

        edges = {i: [] for i in range(4)}
        invalid_edges = set()
        for a, b in contacts:
            lower, upper = (a, b) if positions[a][2] < positions[b][2] else (b, a)
            r = rotations[upper][:2, :2]
            s = self.half_sizes[upper][:2]
            polygon = [positions[upper][:2] + np.array([x, y]) @ r.T
                       for x, y in [(-s[0], -s[1]), (s[0], -s[1]), (s[0], s[1]), (-s[0], s[1])]]
            fraction = self._clip(polygon, positions[lower][:2], rotations[lower][:2, :2],
                                  self.half_sizes[lower][:2]) / float(4 * s[0] * s[1])
            if (-.002 <= bottoms[upper] - tops[lower] <= .002 and fraction >= .80 - 1e-6):
                edges[lower].append(upper)
            else:
                invalid_edges.update((a, b))
        # Any contact with an external support, moving/tipped block, or side brace
        # invalidates its entire connected component, including indirect robot support.
        bad = foreign | invalid_edges | {i for i in range(4) if not valid[i]}
        pending = list(bad)
        while pending:
            i = pending.pop()
            for neighbor in adjacency[i] - bad:
                bad.add(neighbor)
                pending.append(neighbor)
        best_height, best_chain = 0.0, []

        def visit(chain):
            nonlocal best_height, best_chain
            # Every suffix COM must lie inside its supporting block's footprint.
            for k, lower in enumerate(chain[:-1]):
                com = np.mean([positions[i][:2] for i in chain[k+1:]], axis=0)
                local = (com - positions[lower][:2]) @ rotations[lower][:2, :2]
                if np.any(np.abs(local) > self.half_sizes[lower][:2] - .002):
                    return
            height = float(tops[chain[-1]])
            if len(chain) >= 2 and height > best_height:
                best_height, best_chain = height, list(chain)
            for upper in edges[chain[-1]]:
                if upper not in chain and upper not in bad:
                    visit(chain + [upper])

        for bottom in table - bad:
            visit([bottom])
        success = len(best_chain) == 4
        return {"success": success,
                "score": 100.0 if success else min(99.0, 100 * best_height / self.maximum_height),
                "metrics": {"stack_height": best_height, "maximum_height": self.maximum_height,
                            "stacked_blocks": len(best_chain), "invalid_blocks": sorted(bad),
                            "table_supported_blocks": sorted(table)}}
