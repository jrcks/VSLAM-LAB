import os.path

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class VGGT_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='vggt', baseline_folder='vggt'):

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
        print(f"Default executable: vslamlab_vggt")
        
    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        return super().execute(command, exp_it, exp_folder, timeout_seconds)

class VGGT_baseline_dev(VGGT_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'vggt-dev', baseline_folder =  'VGGT-DEV')
        self.color = 'green'

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, 'vslamlab_vggt.py'))
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: {os.path.join(self.baseline_path, 'vslamlab_vggt.py')}")
