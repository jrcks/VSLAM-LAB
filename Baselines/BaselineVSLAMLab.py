import subprocess

from utilities import ws
import os

class BaselineVSLAMLab:

    def __init__(self, baseline_name, baselines_path):
        self.baseline_name = baseline_name
        self.baselines_path = baselines_path
        self.default_parameters = ''
        self.settings_yaml = ''

    def get_default_parameters(self):
        return self.default_parameters

    def execute(self, command, exp_it, exp_folder):
        log_file_path = os.path.join(exp_folder, "system_output_" + str(exp_it).zfill(5) + ".txt")

        with open(log_file_path, 'w') as log_file:
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(command, stdout=log_file, stderr=log_file, shell=True)

    def build_execute_command(self, exp_it, exp, dataset, sequence_name):
        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        exec_command = [f"sequence_path:{sequence_path}", f"exp_folder:{exp_folder}", f"exp_id:{exp_it}"]

        i_par = 0
        for parameter in exp.parameters:
            exec_command += [str(parameter)]
            i_par += 1
        command_str = ' '.join(exec_command)

        full_command = f"pixi run -e {self.baseline_name} execute " + command_str

        return full_command

    def build_execute_command_python(self, exp_it, exp, dataset, sequence_name):
        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        rgb_txt = os.path.join(sequence_path, 'rgb.txt')

        exec_command = [f"--sequence_path {sequence_path}",
                        f"--calibration_yaml {calibration_yaml}",
                        f"--rgb_txt {rgb_txt}",
                        f"--exp_folder {exp_folder}",
                        f"--exp_it {exp_it}",
                        f"--settings_yaml {self.settings_yaml}"]

        return exec_command