import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab
from path_constants import VSLAMLAB_BASELINES


class DROIDSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baselines_path):
        baseline_name = 'droidslam'
        baseline_folder = 'DROID-SLAM'
        default_parameters = {'verbose': 1, 'upsample': 0}

        # Initialize the baseline
        super().__init__(baseline_name, baselines_path)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        self.default_parameters = default_parameters
        self.settings_yaml = os.path.join(VSLAMLAB_BASELINES, baseline_folder, f'{baseline_name}_settings.yaml')

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

        exec_command = []
        verbose = self.default_parameters['verbose']
        if 'verbose' in exp.parameters:
            verbose = exp.parameters['verbose']
        if not verbose:
            exec_command += [f'--disable_vis']

        upsample = self.default_parameters['upsample']
        if 'upsample' in exp.parameters:
            upsample = exp.parameters['upsample']
        if upsample:
            exec_command += [f'--upsample']

        command_str = ' '.join(exec_command)
        vslamlab_command = vslamlab_command + " " + command_str

        return vslamlab_command
