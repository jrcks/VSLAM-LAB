import os.path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class MAST3RSLAM_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'mast3rslam'
        baseline_folder = 'MASt3R-SLAM'
        default_parameters = {'verbose': 1}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'orange'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)        
        return vslamlab_command

    def is_installed(self):
        return True

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/DROID-SLAM/droidslam_vslamlab_mono.py")