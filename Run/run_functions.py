# Run methods

import time
import os
import shutil

from utilities import ws
from Baselines.baseline_utilities import log_run_sequence_time
from Baselines.baseline_utilities import append_ablation_parameters_to_csv

from Run import ablations
from path_constants import ABLATION_PARAMETERS_CSV

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


def run_sequence(exp_it, exp, baseline, dataset, sequence_name, ablation=False):
    print(f"{SCRIPT_LABEL}Running (it {exp_it + 1}/{exp.num_runs}) {baseline.label} in {dataset.dataset_color}{sequence_name}\033[0m of {dataset.dataset_label} ...")
    run_time_start = time.time()

    # Create experiment folder
    exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
    if not os.path.exists(exp_folder):
        os.makedirs(exp_folder, exist_ok=True)

    # Build execution command
    exec_command = baseline.build_execute_command(exp_it, exp, baseline, dataset, sequence_name)

    # Prepare Ablation
    if ablation:
        settings_ablation_yaml, full_command = prepare_ablation(sequence_name, exp, exp_it, exp_folder, dataset, exec_command, baseline)

    # Execute experiment
    baseline.execute(exec_command, exp_it, exp_folder)

    # Finish Ablation
    if ablation:
        finish_ablation(sequence_name, settings_ablation_yaml, dataset)

    # Log iteration duration
    duration_time = time.time() - run_time_start
    log_run_sequence_time(exp_folder, exp_it, duration_time)

    return duration_time

####################################################################################################################

# Ablation methods
def prepare_ablation(sequence_name, exp, it, exp_folder, dataset, full_command, baseline):
    print(f"{ws(8)}Sequence '{sequence_name}' preparing ablation ...")
    settings_yaml = baseline.settings_yaml
    for parameter in exp.parameters:
        if 'settings_yaml' in parameter:
            settings_yaml = parameter.replace('settings_yaml:', '')
        if 'ablation_param' in parameter:
            ablation_param = parameter.replace('ablation_param:', '')

    ablation_parameters = {}
    ablation_parameters['expId'] = str(it).zfill(5)
    ablation_parameters_csv = os.path.join(exp_folder, ABLATION_PARAMETERS_CSV)

    # Define your ablations
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)

    ablation_parameters_i = ablations.add_noise_to_images_start(sequence_path, it, exp, dataset.rgb_hz)
    ablation_parameters.update(ablation_parameters_i)

    settings_ablation_yaml = settings_yaml.replace('_settings', '_settings_ablation')
    if os.path.exists(settings_ablation_yaml):
        os.remove(settings_ablation_yaml)
    shutil.copy(settings_yaml, settings_ablation_yaml)

    ablation_parameters_i = ablations.parameter_ablation_start(it, ablation_param, settings_ablation_yaml)
    ablation_parameters.update(ablation_parameters_i)

    append_ablation_parameters_to_csv(ablation_parameters_csv, ablation_parameters)
    modified_command = full_command.replace(settings_yaml, settings_ablation_yaml)

    return settings_ablation_yaml, modified_command


def finish_ablation(sequence_name, settings_ablation_yaml, dataset):
    print(f"{ws(8)}Sequence '{sequence_name}' finishing ablation ...")
    sequence_path = os.path.join(dataset.dataset_path, sequence_name)
    ablations.add_noise_to_images_finish(sequence_path)
    ablations.parameter_ablation_finish(settings_ablation_yaml)
