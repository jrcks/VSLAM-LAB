import os.path
import yaml

from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

from path_constants import VSLAMLAB_BASELINES


class DROIDSLAM_baseline(BaselineVSLAMLab):
    def __init__(self, baselines_path):
        baseline_name = 'droidslam'
        baseline_folder = 'DROID-SLAM'
        default_parameters = ['verbose:1', 'upsample:0']

        # Initialize the baseline
        super().__init__(baseline_name, baselines_path)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        self.default_parameters = default_parameters

    def build_execute_command(self, sequence_path, exp_folder, exp_it, parameters):

        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')
        exec_command = [f"--sequence_path {sequence_path}", f"--calibration_yaml {calibration_yaml}",
                        f"--rgb_txt {rgb_txt}",
                        f"--exp_folder {exp_folder}", f"--exp_it {exp_it}"]

        settings_yaml = 'Baselines/DROID-SLAM/droidslam_settings.yaml'
        with open(settings_yaml, 'r') as file:
            settings = yaml.safe_load(file)

        for parameter in parameters:
            parameter_name = parameter.split(":")[0]
            parameter_value = str(parameter.split(":")[1])
            if 'verbose' in parameter_name:
                verbose = bool(int((parameter.replace('verbose:', ''))))
                if not verbose:
                    exec_command += [f'--disable_vis']
                continue
            if 'upsample' in parameter_name:
                upsample = parameter.replace('upsample:', '')
                if upsample:
                    exec_command += [f'--upsample']
                continue

            settings[parameter_name] = parameter_value

        for setting in settings:
            exec_command += [f'--{setting}', str(settings[f'{setting}'])]

        command_str = ' '.join(exec_command)
        full_command = f"pixi run -e {self.baseline_name} execute " + command_str

        return full_command
