import os.path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class DROIDSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='droidslam', baseline_folder='DROID-SLAM'):

        default_parameters = {'verbose': 1, 'upsample': 0, 'mode': 'mono', 'weights': 'droid.pth'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'green'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)
        
        # Add the mode argument
        mode = self.default_parameters['mode']
        if 'mode' in exp.parameters:
            mode = exp.parameters['mode']

        if mode == "mono":
            vslamlab_command = vslamlab_command.replace('execute', 'execute_mono')

        # Add the weights argument
        weights = self.default_parameters['weights']
        if 'weights' in exp.parameters:
            weights = exp.parameters['weights']
        vslamlab_command += ' ' + '--weights ' + os.path.join(self.baseline_path, weights)
        
        return vslamlab_command

    def is_cloned(self):
        return True
    
    def is_installed(self): 
        return self.droidslam_download_weights() and self.download_vslamlab_settings()
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: droidslam_vslamlab_mono")
        
    def droidslam_download_weights(self): # Download droid.pth
        weights_pth = os.path.join(self.baseline_path, 'droid.pth')
        if not os.path.exists(weights_pth):
            print_msg(SCRIPT_LABEL, "Downloading droid.pth ...",'info')
            file_path = hf_hub_download(repo_id='vslamlab/droidslam', filename='droid.pth', repo_type='model',
                                        local_dir=self.baseline_path)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.baseline_path)
        return os.path.isfile(weights_pth)
                
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
