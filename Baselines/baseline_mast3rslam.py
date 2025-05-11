import os.path
from huggingface_hub import hf_hub_download

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class MAST3RSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='mast3rslam', baseline_folder='MASt3R-SLAM'):
        
        default_parameters = {'verbose': 1, 'checkpoints_dir': 'checkpoints'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'orange'

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        vslamlab_command = super().build_execute_command_python(exp_it, exp, dataset, sequence_name)        

        if 'calib' in exp.parameters:
            if not exp.parameters['calib']:
                vslamlab_command += ' ' + '--no_calib '

        checkpoints_dir = self.default_parameters['checkpoints_dir']
        if 'checkpoints_dir' in exp.parameters:
            checkpoints_dir = exp.parameters['checkpoints_dir']
        vslamlab_command += ' ' + '--checkpoints_dir ' + os.path.join(self.baseline_path, checkpoints_dir)

        return vslamlab_command

    def is_cloned(self):
        return True
    
    def is_installed(self):
        return True
    
    def info_print(self):
        super().info_print()
        print(f"Default executable: vslamlab_mast3rslam_mono")

    def mast3rslam_download_weights(self): # Download checkpoints
        checkpoints_dir = os.path.join(self.baseline_path,'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        files = [
            os.path.join(checkpoints_dir, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"),
            os.path.join(checkpoints_dir, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth"),
            os.path.join(checkpoints_dir, "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl")
        ]

        for file in files:
            file_name = os.path.basename(file)
            if not os.path.exists(file):
                print_msg(SCRIPT_LABEL, f"Downloading {file}",'info')
                _ = hf_hub_download(repo_id='vslamlab/mast3rslam', filename=file_name, repo_type='model', local_dir=checkpoints_dir)  

        return os.path.exists(files[0]) and os.path.exists(files[1]) and os.path.exists(files[2])
    
    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        self.mast3rslam_download_weights() 
        self.download_vslamlab_settings()
        return super().execute(command, exp_it, exp_folder, timeout_seconds)

class MAST3RSLAM_baseline_dev(MAST3RSLAM_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'mast3rslam-dev', baseline_folder =  'MASt3R-SLAM-DEV')
        self.color = 'orange'

    def is_cloned(self):
        return os.path.isdir(os.path.join(self.baseline_path, '.git'))

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, f'install_{self.baseline_name}.txt'))

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/MASt3R-SLAM/vslamlab_mast3rslam_mono.py")