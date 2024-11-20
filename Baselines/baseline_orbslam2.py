import os.path

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab
from path_constants import VSLAMLAB_BASELINES


class ORBSLAM2_baseline(BaselineVSLAMLab):
    def __init__(self, baselines_path):
        baseline_name = 'orbslam2'
        baseline_folder = 'ORB_SLAM2_VSLAMLAB'
        baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        default_parameters = {'verbose': 1,
                              'vocabulary': os.path.join(baseline_path, 'Vocabulary', 'ORBvoc.txt'),
                              'mode': 'mono'}

        # Initialize the baseline
        super().__init__(baseline_name, baselines_path)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.baseline_path = baseline_path
        self.default_parameters = default_parameters
        self.settings_yaml = os.path.join(VSLAMLAB_BASELINES, baseline_folder, f'{baseline_name}_settings.yaml')

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):

        vslamlab_command = super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

        mode = self.default_parameters['mode']
        if 'mode' in exp.parameters:
            mode = exp.parameters['mode']

        if mode == "mono":
            vslamlab_command = f"pixi run -e {self.baseline_name} execute_mono " + ' '.join(vslamlab_command)
        if mode == "rgbd":
            vslamlab_command = f"pixi run -e {self.baseline_name} execute_rgbd " + ' '.join(vslamlab_command)

        return vslamlab_command