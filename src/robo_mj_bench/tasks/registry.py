"""Fixed trusted task registry. Caller input is never a module or file path."""

from importlib import import_module

TASKS = {
    "placement": ("placement", "PlacementTask"),
    "pouring": ("pouring", "PouringTask"),
    "tidying": ("tidying", "TidyingTask"),
    "shape_matching": ("shape_matching", "ShapeMatchingTask"),
    "writing": ("writing", "WritingTask"),
    "tower": ("tower", "TowerTask"),
    "measurement": ("measurement", "MeasurementTask"),
    "bimanual_handover": ("bimanual_tasks", "HandoverTask"),
    "bimanual_tray": ("bimanual_tasks", "TrayCarryTask"),
    "bimanual_drawer": ("bimanual_tasks", "DrawerInsertTask"),
}
BIMANUAL_TASKS = ("bimanual_handover", "bimanual_tray", "bimanual_drawer")
TASK_NAMES = ("button", *TASKS)


def create_task(name):
    if name not in TASKS:
        raise ValueError("Unknown task")
    module, cls = TASKS[name]
    return getattr(import_module(f"robo_mj_bench.tasks.{module}"), cls)()


def create_simulation(name, seed):
    if name == "button":
        from ..simulation import ButtonSimulation
        return ButtonSimulation(seed)
    from ..task_simulation import TaskSimulation
    if name in BIMANUAL_TASKS:
        from ..bimanual import BimanualSimulation
        return BimanualSimulation(create_task(name), seed)
    return TaskSimulation(create_task(name), seed)
