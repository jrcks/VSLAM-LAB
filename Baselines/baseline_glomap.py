import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

from path_constants import VSLAMLAB_BASELINES


class GLOMAP_baseline(BaselineVSLAMLab):
    def __init__(self, baselines_path):
        baseline_name = 'glomap'
        baseline_folder = 'glomap'
        default_parameters = {'verbose': 1, 'matcher_type': 'sequential', 'use_gpu': 1}

        # Initialize the baseline
        super().__init__(baseline_name, baselines_path)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        self.default_parameters = default_parameters
        self.settings_yaml = os.path.join(VSLAMLAB_BASELINES, baseline_folder, f'{baseline_name}_settings.yaml')

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

        exec_command = []
        for parameter_name, parameter_value in self.default_parameters.items():
            if parameter_name in exp.parameters:
                exec_command += [f"{str(parameter_name)}:{str(exp.parameters[parameter_name])}"]
            else:
                exec_command += [f"{str(parameter_name)}:{str(parameter_value)}"]

        command_str = ' '.join(exec_command)
        vslamlab_command = vslamlab_command + " " + command_str

        return vslamlab_command
