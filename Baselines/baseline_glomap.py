import os.path

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab
from Baselines.baseline_colmap import COLMAP_baseline

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class GLOMAP_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='glomap', baseline_folder='glomap'):

        default_parameters = {'verbose': 1, 'matcher_type': 'exhaustive', 'use_gpu': 1, 'max_rgb': 200}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))
    
    def git_clone(self):
        colmap = COLMAP_baseline()
        if not colmap.is_cloned():
            colmap.git_clone()

        super().git_clone()

    def is_installed(self): 
        return True

    def info_print(self):
        super().info_print()
        print(f"Default executable: {os.path.join(self.baseline_path, 'glomap_reconstruction.sh')}")