import os.path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class MONOGS_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='monogs', baseline_folder='MonoGS'):

        default_parameters = {'verbose': 1}    
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'yellow'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        return vslamlab_command

    def is_cloned(self):
        return True
    
    def is_installed(self):
        return True
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: slam.py")

    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        self.download_vslamlab_settings()
        super().execute(command, exp_it, exp_folder, timeout_seconds)
        
class MONOGS_baseline_dev(MONOGS_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'monogs-dev', baseline_folder =  'MonoGS-DEV')
        self.color = 'red'

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))

    def is_installed(self):
        return (os.path.isdir(os.path.join(self.baseline_path, 'submodules', 'simple-knn', 'build')) and
                os.path.isdir(os.path.join(self.baseline_path, 'submodules', 'diff-gaussian-rasterization', 'build')))
        
    def info_print(self):
        super().info_print()
        print(f"Default executable: {self.baseline_path}/vslamlab_{self.baseline_name}_mono.py")