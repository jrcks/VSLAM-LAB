import os.path
import yaml
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class GLOMAP_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'glomap'
        baseline_folder = 'glomap'
        default_parameters = {'verbose': 1, 'matcher_type': 'sequential', 'use_gpu': 1, 'max_rgb': 100}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, 'bin', 'glomap'))

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/glomap/bin/glomap")

    def modify_yaml_parameter(self, yaml_file, section_name, parameter_name, new_value):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        if section_name in data and parameter_name in data[section_name]:
            data[section_name][parameter_name] = new_value
        else:
            print(f"    Parameter '{parameter_name}' or section '{section_name}' not found in the YAML file.")

        with open(yaml_file, 'w') as file:
            yaml.safe_dump(data, file)
