import os.path

from utilities import print_msg
from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class GENSFM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='gensfm', baseline_folder='GenSfM'):
        default_parameters = {'verbose': 1, 'max_rgb': 200}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'blue'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))
    
    def is_installed(self):
        return True
    
class GENSFM_baseline_dev(GENSFM_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'gensfm-dev', baseline_folder = 'GenSfM-DEV')
        self.color = 'blue'

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, 'build', 'src', 'exe', 'gen_colmap'))
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: {os.path.isfile(os.path.join(self.baseline_path, 'build', 'src', 'exe', 'gen_colmap'))}")

    
    
