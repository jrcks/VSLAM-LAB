import os.path

from path_constants import VSLAMLAB_BASELINES
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class DROIDSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='droidslam', baseline_folder='DROID-SLAM'):

        default_parameters = {'verbose': 1, 'mode': 'mono', 'upsample': 0, 
                              'weights': f'{os.path.join(VSLAMLAB_BASELINES, baseline_folder, 'droid.pth')}'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'
        self.modes = ['mono']

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
    def is_installed(self): 
        return True, f'conda package available'
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: droidslam_vslamlab_mono")
        
    
class DROIDSLAM_baseline_dev(DROIDSLAM_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'droidslam-dev', baseline_folder =  'DROID-SLAM-DEV')
        self.color = 'green'

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, f'install_{self.baseline_name}.txt'))
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/DROID-SLAM/droidslam_vslamlab_mono.py")
