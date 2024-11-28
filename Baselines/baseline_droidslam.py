import os.path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class DROIDSLAM_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'droidslam'
        baseline_folder = 'DROID-SLAM'
        default_parameters = {'verbose': 1, 'upsample': 0}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, 'droid.pth'))

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/DROID-SLAM/droidslam_vslamlab.py")