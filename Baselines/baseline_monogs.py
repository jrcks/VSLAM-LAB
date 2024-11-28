import os.path
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab


class MONOGS_baseline(BaselineVSLAMLab):
    def __init__(self):
        baseline_name = 'monogs'
        baseline_folder = 'MonoGS'

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder)

        self.default_parameters = {'verbose': 1, 'config_yaml': f'{os.path.join(self.baseline_path, 'configs', 'mono', 'vslamlab', 'vslamlab.yaml')}'}


    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def is_installed(self):
        return (os.path.isdir(os.path.join(self.baseline_path, 'submodules', 'simple-knn', 'build')) and
                os.path.isdir(os.path.join(self.baseline_path, 'submodules', 'diff-gaussian-rasterization', 'build')))

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/MonoGS/slam.py")
