import os.path

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class MAST3R_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='mast3r', baseline_folder='mast3r'):

        default_parameters = {'verbose': 1, 'max_rgb': 20}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_cloned(self):
        return True
    
    def is_installed(self): 
        return True
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: vslamlab_mast3r_demo.py")
        
class MAST3R_baseline_dev(MAST3R_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'mast3r-dev', baseline_folder =  'MASt3R-DEV')
        self.color = 'green'

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))

    def is_installed(self):
        return os.path.isdir(os.path.join(self.baseline_path, 'asmk', 'build')) and os.path.isdir(os.path.join(
            self.baseline_path, 'dust3r', 'croco', 'models', 'curope', 'build'))
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: {os.path.join(self.baseline_path, 'vslamlab_mast3r_demo.py')}")
