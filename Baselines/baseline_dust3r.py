import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

from path_constants import VSLAMLAB_BASELINES
class DUST3R_baseline(BaselineVSLAMLab):
    def __init__(self, baselines_path):
        baseline_name = 'dust3r'
        baseline_folder = 'dust3r'
        baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        default_parameters = {'verbose': 1, 'max_rgb': 10}

        # Initialize the baseline
        super().__init__(baseline_name, baselines_path)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.baseline_path = baseline_path
        self.default_parameters = default_parameters
        self.settings_yaml = os.path.join(VSLAMLAB_BASELINES, baseline_folder, f'{baseline_name}_settings.yaml')

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

        exec_command = []
        verbose = self.default_parameters['verbose']
        if 'verbose' in exp.parameters:
            verbose = exp.parameters['verbose']
        if verbose:
            exec_command += [f'--verbose']

        command_str = ' '.join(exec_command)
        vslamlab_command = vslamlab_command + " " + command_str

        return vslamlab_command
