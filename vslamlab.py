"""
Module: VSLAM-LAB - vslamlab.py
- Author: Alejandro Fontan Villacampa
- Version: 1.0
- Created: 2024-07-04
- Updated: 2024-07-04
- License: GPLv3 License
- List of Known Dependencies;
    * ...
"""

import argparse
import glob
import os
import sys
import time
import shutil
import re
import yaml

from Compare import compare_functions
from Datasets.dataset_utilities import get_dataset
from utilities import COMPARISONS_YAML_DEFAULT
from utilities import CONFIG_DEFAULT
from utilities import EXP_YAML_DEFAULT
from utilities import Experiment
from utilities import VSLAMLAB_BENCHMARK
from utilities import VSLAMLAB_EVALUATION
from utilities import VSLAM_LAB_DIR
from utilities import check_yaml_file_integrity
from utilities import list_datasets
from utilities import ws

SCRIPT_LABEL = f"[{os.path.basename(__file__)}] "


def main():
    # Parse inputs
    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--exp_yaml', nargs='?', type=str,
                        const=EXP_YAML_DEFAULT, default=EXP_YAML_DEFAULT,
                        help=f"Path to the YAML file containing the list of experiments. "
                             f"Default \'vslamlab --exp_yaml {EXP_YAML_DEFAULT}\'")

    parser.add_argument('-download', action='store_true', help="If none 'download/run/evaluate/compare' are activated, "
                                                               "all work by default.")
    parser.add_argument('-run', action='store_true', help="")
    parser.add_argument('-evaluate', action='store_true', help="")
    parser.add_argument('-compare', action='store_true', help="")

    parser.add_argument('--list_datasets', action='store_true', help="List available datasets.")

    args = parser.parse_args()

    if not os.path.exists(VSLAMLAB_EVALUATION):
        os.makedirs(VSLAMLAB_EVALUATION, exist_ok=True)

    # Stuff to run demo
    if not os.path.exists(os.path.join(VSLAMLAB_EVALUATION, "exp_demo_dso")):
        shutil.copytree(os.path.join(VSLAM_LAB_DIR, "docs", "exp_demo_dso"),
                    os.path.join(VSLAMLAB_EVALUATION, "exp_demo_dso"))
    if not os.path.exists(os.path.join(VSLAMLAB_EVALUATION, "exp_demo_orbslam2")):
        shutil.copytree(os.path.join(VSLAM_LAB_DIR, "docs", "exp_demo_orbslam2"),
                    os.path.join(VSLAMLAB_EVALUATION, "exp_demo_orbslam2"))

    print(f"\n[vslamlab] Created folder to store data: {VSLAMLAB_BENCHMARK}")
    print(f"[vslamlab] Created folder to store evaluation: {VSLAMLAB_EVALUATION}")

    # Info commands
    if args.list_datasets:
        print_datasets()
        return

    # Load experiment info
    experiments, config_files = load_experiments(args.exp_yaml)
    check_config_integrity(config_files)

    # Process experiments
    filter_inputs(args)
    if args.download:
        download(config_files)

    if args.run:
        run(experiments)

    if args.evaluate:
        evaluate(experiments)

    if args.compare:
        compare(experiments, args.exp_yaml)


def filter_inputs(args):
    if not args.download and not args.run and not args.evaluate and not args.compare:
        args.download = True
        args.run = True
        args.evaluate = True
        args.compare = True


def check_parameter_for_relative_path(parameter_value):
    if "VSLAM-LAB" in parameter_value:
        if ":" in parameter_value:
            return re.sub(r'(?<=:)[^:]*VSLAM-LAB', VSLAM_LAB_DIR, str(parameter_value))
        return re.sub(r'^.*VSLAM-LAB', VSLAM_LAB_DIR, str(parameter_value))
    return parameter_value


def load_experiments(exp_yaml):
    """
    Loads experiment configurations from a YAML file and initializes Experiment objects.

    Parameters
    ----------
    exp_yaml : str
        Path to the YAML file containing experiment settings (default: VSLAM-LAB/docs/experimentList.yaml).

    Returns
    ----------
    experiments : dict
        experiments<exp_name,Experiment()>
    config_files : dict
        config_files<config_yaml,False>
    """

    check_yaml_file_integrity(exp_yaml)

    with open(exp_yaml, 'r') as file:
        experiment_data = yaml.safe_load(file)

    experiments = {}
    config_files = {}
    for exp_name, settings in experiment_data.items():
        experiment = Experiment()
        active = settings.get('Active', True)
        if not active:
            continue

        experiments[exp_name] = experiment
        experiments[exp_name].config_yaml = os.path.join(VSLAM_LAB_DIR, 'configs',
                                                         settings.get('Config', CONFIG_DEFAULT))
        config_files[experiments[exp_name].config_yaml] = False
        experiments[exp_name].folder = os.path.join(VSLAMLAB_EVALUATION, exp_name)
        experiments[exp_name].num_runs = settings.get('NumRuns', 1)
        experiments[exp_name].vslam = settings.get('VSLAM', "default")

        if settings['Parameters']:
            for parameter_name in settings['Parameters']:
                experiments[exp_name].parameters.append(
                    check_parameter_for_relative_path(settings['Parameters'][parameter_name]))

    print(f"\n[vslamlab] Experiment summary: {os.path.basename(exp_yaml)}")
    print(f"{ws(4)} Number of experiments: {len(experiments)}")
    #print(f"{ws(4)} Estimated data size: - ")
    #run_time = estimate_experiments_time(experiments)
    #if run_time > 60.0:
    #    print(f"{ws(4)} Estimated running time: {run_time/60.0} (h)")
    #else:
    #    print(f"{ws(4)} Estimated running time: {run_time} (min)")

    return experiments, config_files


def compare(experiments, exp_yaml):

    comparison_path = os.path.join(VSLAMLAB_EVALUATION, f"comp_{str(os.path.basename(exp_yaml)).replace('.yaml', '')}")
    print(f"\n[vslamlab] Create folder to save comparison: {comparison_path}")
    print(f"\n[vslamlab] Comparing (in {comparison_path}) ...")
    if os.path.exists(comparison_path):
        shutil.rmtree(comparison_path)
    os.makedirs(comparison_path)
    os.makedirs(os.path.join(comparison_path, 'figures'))
    compare_functions.full_comparison(experiments, VSLAMLAB_BENCHMARK, COMPARISONS_YAML_DEFAULT, comparison_path)


def evaluate(experiments):
    print(f"\n[vslamlab] Evaluating (in {VSLAMLAB_EVALUATION}) ...")
    for [_, exp] in experiments.items():
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    dataset.evaluate_sequence(sequence_name, exp.folder)


def run(experiments):
    print(f"\n[vslamlab] Running ...")
    start_time = time.time()
    for [exp_name, exp] in experiments.items():
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for it in range(exp.num_runs):
                for dataset_name, sequence_names in config_file_data.items():
                    dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                    for sequence_name in sequence_names:
                        sequence_folder = os.path.join(exp.folder, dataset_name.upper(), sequence_name)
                        num_system_output_files = 0
                        if os.path.exists(sequence_folder):
                            search_pattern = os.path.join(sequence_folder, f'*system_output_*')
                            num_system_output_files = len(glob.glob(search_pattern))
                        if num_system_output_files < exp.num_runs:
                            print(
                                f"{ws(4)}Running (it: {num_system_output_files + 1}/{exp.num_runs}) '{exp.vslam}' "
                                f"in: '{sequence_name}'...")
                            dataset.run_sequence(exp, sequence_name)
        if num_system_output_files == exp.num_runs:
            print(f"{ws(4)}Finished experiment '{exp_name}' with {num_system_output_files}/{exp.num_runs} iterations.")
    end_time = time.time()

    run_time = (end_time - start_time) / 60.0
    if run_time > 60.0:
        print(f"\n[vslamlab] Experiment runtime: {run_time / 60.0} (h)")
    else:
        print(f"\n[vslamlab] Experiment runtime: {run_time} (min)")


def download(config_files):
    print(f"\n[vslamlab] Downloading (to {VSLAMLAB_BENCHMARK}) ...")
    for config_file in config_files:
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    dataset.download_sequence(sequence_name)


def check_config_integrity(config_files):
    dataset_list = list_datasets()
    for config_file in config_files:
        check_yaml_file_integrity(config_file)
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                if not (dataset_name in dataset_list):
                    print(f"\n{SCRIPT_LABEL}Error in : {config_file}")
                    print(f"{ws(4)}'{dataset_name}' dataset doesn't exist")
                    print_datasets()
                    sys.exit(1)

                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    if not dataset.contains_sequence(sequence_name):
                        print(f"\n{SCRIPT_LABEL}Error in : {config_file}")
                        print(f"{ws(4)}'{dataset_name}' dataset doesn't contain sequence '{sequence_name}'")
                        print(f"\nAvailable sequences are: {dataset.sequence_names}")
                        print(f"")
                        sys.exit(1)


def print_datasets():
    dataset_list = list_datasets()
    print(f"\n{SCRIPT_LABEL}Accessible datasets in VSLAM-LAB:")
    for dataset in dataset_list:
        print(f" - {dataset}")
    print("")


def estimate_experiments_time(experiments):
    running_time = 0
    for [exp_name, exp] in experiments.items():
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    num_frames = dataset.get_sequence_num_rgb(sequence_name)
                    running_time += exp.num_runs * num_frames / dataset.rgb_hz
    return 1.5 * running_time / 60


if __name__ == "__main__":
    main()
