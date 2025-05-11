import os.path
import stat
from huggingface_hub import hf_hub_download

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class GLOMAP_baseline(BaselineVSLAMLab):
    def __init__(self, baseline_name='glomap', baseline_folder='glomap'):

        default_parameters = {'verbose': 1, 'matcher_type': 'sequential', 'use_gpu': 1}

        # Initialize the baseline
        super().__init__(baseline_name, baseline_folder, default_parameters)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        return super().build_execute_command_cpp(exp_it, exp, dataset, sequence_name)

    def is_cloned(self):
        return True
    
    def is_installed(self): 
        return True

    def info_print(self):
        super().info_print()
        print(f"Default executable: ")

    def glomap_download_bag_of_words(self):
        files = [
            os.path.join(self.baseline_path, "vocab_tree_flickr100K_words1M.bin"),
            os.path.join(self.baseline_path, "vocab_tree_flickr100K_words32K.bin"),
            os.path.join(self.baseline_path, "vocab_tree_flickr100K_words256K.bin")
        ]

        for file in files:
            file_name = os.path.basename(file)
            if not os.path.exists(file):
                print_msg(SCRIPT_LABEL, f"Downloading {file}",'info')
                _ = hf_hub_download(repo_id='vslamlab/glomap', filename=file_name, repo_type='model', local_dir=self.baseline_path)  

        return os.path.exists(files[0]) and os.path.exists(files[1]) and os.path.exists(files[2])
    
    def glomap_download_executables(self):
        files = [
            os.path.join(self.baseline_path, "colmap_matcher.sh"),
            os.path.join(self.baseline_path, "glomap_mapper.sh"),
            os.path.join(self.baseline_path, "glomap_reconstruction.sh"),
            os.path.join(self.baseline_path, "colmap_to_vslamlab.py")
        ]

        for file in files:
            file_name = os.path.basename(file)
            if not os.path.exists(file):
                print_msg(SCRIPT_LABEL, f"Downloading {file}",'info')
                _ = hf_hub_download(repo_id='vslamlab/glomap', filename=file_name, repo_type='model', local_dir=self.baseline_path)  

            if os.path.exists(file):
                os.chmod(file, os.stat(file).st_mode | stat.S_IEXEC)
        return os.path.exists(files[0]) and os.path.exists(files[1]) and os.path.exists(files[2])
    
    def execute(self, command, exp_it, exp_folder, timeout_seconds=1*60*10):
        self.download_vslamlab_settings()
        self.glomap_download_bag_of_words()
        self.glomap_download_executables()
        return super().execute(command, exp_it, exp_folder, timeout_seconds)

# class GLOMAP_baseline_dev(GLOMAP_baseline):
#     def __init__(self):
#         super().__init__(baseline_name = 'glomap-dev', baseline_folder =  'glomap-DEV')

#     def is_cloned(self):
#         return os.path.isdir(os.path.join(self.baseline_path, '.git'))

#     def is_installed(self):
#         return os.path.isfile(os.path.join(self.baseline_path, 'bin', 'glomap'))
    
#     def info_print(self):
#         super().info_print()
#         print(f"Default executable: Baselines/glomap/bin/glomap")
