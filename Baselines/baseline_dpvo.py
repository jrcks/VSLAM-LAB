import os.path
from zipfile import ZipFile
from huggingface_hub import hf_hub_download

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class DPVO_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='dpvo', baseline_folder='DPVO'):
       
        default_parameters = {'verbose': 1, 'network': 'dpvo.pth'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'red'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

        network = self.default_parameters['network']
        if 'network' in exp.parameters:
            network = exp.parameters['network']
        vslamlab_command += ' ' + '--network ' + os.path.join(self.baseline_path, network)

        return vslamlab_command

    def is_cloned(self):
        return True
    
    def is_installed(self):
        return True

    def info_print(self):
        super().info_print()
        print(f"Default executable: vslamlab_{self.baseline_name}_mono")

    def dpvo_download_weights(self): # Download dpvo.pth
        weights_pth = os.path.join(self.baseline_path, 'dpvo.pth')
        if not os.path.exists(weights_pth):
            print_msg(SCRIPT_LABEL, "Downloading dpvo.pth ...",'info')
            file_path = hf_hub_download(repo_id='vslamlab/dpvo', filename='models.zip', repo_type='model',
                                        local_dir=self.baseline_path)
            with ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(self.baseline_path)
        return os.path.isfile(weights_pth)
    
    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        self.dpvo_download_weights() 
        self.download_vslamlab_settings()
        super().execute(command, exp_it, exp_folder, timeout_seconds)

class DPVO_baseline_dev(DPVO_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'dpvo-dev', baseline_folder =  'DPVO-DEV')
        self.color = 'red'

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, f'install_{self.baseline_name}.txt'))
        
    def info_print(self):
        super().info_print()
        print(f"Default executable: {self.baseline_path}/vslamlab_{self.baseline_name}_mono.py")
