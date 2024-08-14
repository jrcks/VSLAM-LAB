# ADD your imports here
from Datasets.dataset_7scenes import SEVENSCENES_dataset
from Datasets.dataset_eth import ETH_dataset
from Datasets.dataset_euroc import EUROC_dataset
from Datasets.dataset_kitti import KITTI_dataset
from Datasets.dataset_monotum import MONOTUM_dataset
from Datasets.dataset_nuim import NUIM_dataset
from Datasets.dataset_rgbdtum import RGBDTUM_dataset
from Datasets.dataset_tartanair import TARTANAIR_dataset

SCRIPT_LABEL = "[dataset_utilities.py] "


def get_dataset(dataset_name, benchmark_path):
    dataset_name = dataset_name.lower()
    switcher = {
        # ADD your datasets here
        "rgbdtum": lambda: RGBDTUM_dataset(benchmark_path),
        "eth": lambda: ETH_dataset(benchmark_path),
        "kitti": lambda: KITTI_dataset(benchmark_path),
        "euroc": lambda: EUROC_dataset(benchmark_path),
        "monotum": lambda: MONOTUM_dataset(benchmark_path),
        "nuim": lambda: NUIM_dataset(benchmark_path),
        "7scenes": lambda: SEVENSCENES_dataset(benchmark_path),
        "tartanair": lambda: TARTANAIR_dataset(benchmark_path)
    }

    return switcher.get(dataset_name, lambda: "Invalid case")()
