import os.path
import requests
from tqdm import tqdm
from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class MAST3RSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='mast3rslam', baseline_folder='conda-packages/MASt3R-SLAM'):
        default_parameters = {'verbose': 1, 'checkpoints_dir': 'checkpoints'}
        
        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)
        self.color = 'orange'
        self.name_label = 'MASt3R-SLAM'

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

    def info_print(self):
        super().info_print()
        print(f"Default executable: Baselines/MASt3R-SLAM/vslamlab_mast3rslam_mono.py")

    def mast3rslam_download_weights(self): # Download checkpoints
        checkpoints_dir = os.path.join(self.baseline_path,'checkpoints')
        os.makedirs(checkpoints_dir, exist_ok=True)

        urls = [
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth",
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_trainingfree.pth",
        "https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric_retrieval_codebook.pkl"]

        chunk_size=1024
        for url in urls:
            filename = os.path.join(checkpoints_dir, os.path.basename(url))
            if os.path.exists(filename):
                continue

            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length', 0))
            desc = os.path.basename(filename)

            print(f"Downloading {url} to {filename}")
            with open(filename, 'wb') as f, tqdm(
                total=total, unit='B', unit_scale=True, desc=desc
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        self.mast3rslam_download_weights()
        super().execute(command, exp_it, exp_folder, timeout_seconds)

    def is_cloned(self):
        return True
    
    def is_installed(self):
        return True
    
class MAST3RSLAM_baseline_dev(MAST3RSLAM_baseline):
    def __init__(self):
        super().__init__(baseline_name = 'mast3rslam-dev', baseline_folder =  'MASt3R-SLAM')
        self.color = 'orange'
        self.name_label = 'MASt3R-SLAM-DEV'

    def is_installed(self):
        return os.path.isfile(os.path.join(self.baseline_path, f'install_{self.baseline_name}.txt'))

    def is_cloned(self):
        if os.path.isdir(os.path.join(self.baseline_path, '.git')):
            return True
        return False