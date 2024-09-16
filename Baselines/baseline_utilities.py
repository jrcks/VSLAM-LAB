# ADD your imports here
from Baselines.baseline_anyfeature import ANYFEATURE_baseline
from Baselines.baseline_dso import DSO_baseline
from Baselines.baseline_orbslam2 import ORBSLAM2_baseline
from Baselines.baseline_dust3r import DUST3R_baseline
from Baselines.baseline_monogs import MONOGS_baseline
from Baselines.baseline_colmap import COLMAP_baseline
from Baselines.baseline_glomap import GLOMAP_baseline

SCRIPT_LABEL = "[baseline_utilities.py] "


def get_baseline(baseline_name, baselines_path):
    baseline_name = baseline_name.lower()
    switcher = {
        # ADD your baselines here
        "anyfeature": lambda: ANYFEATURE_baseline(baselines_path),
        "dso": lambda: DSO_baseline(baselines_path),
        "orbslam2": lambda: ORBSLAM2_baseline(baselines_path),
        "dust3r": lambda: DUST3R_baseline(baselines_path),
        "monogs": lambda: MONOGS_baseline(baselines_path),
        "colmap": lambda: COLMAP_baseline(baselines_path),
        "glomap": lambda: GLOMAP_baseline(baselines_path),
    }

    return switcher.get(baseline_name, lambda: "Invalid case")()
