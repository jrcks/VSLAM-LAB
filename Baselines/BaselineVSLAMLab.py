import subprocess

from utilities import ws
import os
from path_constants import VSLAMLAB_BASELINES

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class BaselineVSLAMLab:

    def __init__(self, baseline_name, baseline_folder, default_parameters=''):
        self.baseline_name = baseline_name
        self.baseline_path = os.path.join(VSLAMLAB_BASELINES, baseline_folder)
        self.label = f"\033[96m{baseline_name}\033[0m"
        self.settings_yaml = os.path.join(VSLAMLAB_BASELINES, baseline_folder, f'{baseline_name}_settings.yaml')
        self.default_parameters = default_parameters

    def get_default_parameters(self):
        return self.default_parameters

    def git_clone(self):
        if os.path.isdir(self.baseline_path):
            return

        log_file_path = os.path.join(VSLAMLAB_BASELINES, f'git_clone_{self.baseline_name}.txt')
        git_clone_command = f"pixi run --frozen -e {self.baseline_name} git-clone"
        with open(log_file_path, 'w') as log_file:
            print(f"\n{SCRIPT_LABEL}git clone {self.label}\033[0m : {self.baseline_path}")
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(git_clone_command, shell=True, stdout=log_file, stderr=log_file)

    def is_installed(self):
        return False

    def install(self):
        if self.is_installed():
            return

        log_file_path = os.path.join(self.baseline_path, f'install_{self.baseline_name}.txt')
        install_command = f"pixi run --frozen -e {self.baseline_name} install -v"
        with open(log_file_path, 'w') as log_file:
            print(f"\n{SCRIPT_LABEL}Installing {self.label}\033[0m : {self.baseline_path}")
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(install_command, shell=True, stdout=log_file, stderr=log_file)

    def check_installation(self):
        self.git_clone()
        self.install()

    def info_print(self):
        print(f'Name: {self.label}')
        is_installed = self.is_installed()
        print(f"Installed:\033[92m {is_installed}\033[0m" if is_installed else f"Installed:\033[91m {is_installed}\033[0m")
        print(f"Path:\033[92m {self.baseline_path}\033[0m" if is_installed else f"Path:\033[91m {self.baseline_path}\033[0m")
        print(f'Default parameters: {self.get_default_parameters()}')

    def execute(self, command, exp_it, exp_folder):
        log_file_path = os.path.join(exp_folder, "system_output_" + str(exp_it).zfill(5) + ".txt")
        with open(log_file_path, 'w') as log_file:
            print(f"{ws(6)} log file: {log_file_path}")
            subprocess.run(command, stdout=log_file, stderr=log_file, shell=True)

    def build_execute_command(self, sequence_path, exp_folder, exp_it, parameters):
        exec_command = [f"sequence_path:{sequence_path}", f"exp_folder:{exp_folder}", f"exp_id:{exp_it}"]

        i_par = 0
        for parameter in parameters:
            exec_command += [str(parameter)]
            i_par += 1
        command_str = ' '.join(exec_command)

        full_command = f"pixi run -e {self.baseline_name} execute " + command_str
        return full_command

    def build_execute_command_cpp(self, exp_it, exp, dataset, sequence_name):
        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        rgb_exp_txt = os.path.join(exp_folder, 'rgb_exp.txt')

        vslamlab_command = [f"sequence_path:{sequence_path}",
                            f"calibration_yaml:{calibration_yaml}",
                            f"rgb_txt:{rgb_exp_txt}",
                            f"exp_folder:{exp_folder}",
                            f"exp_id:{exp_it}",
                            f"settings_yaml:{self.settings_yaml}"]

        for parameter_name, parameter_value in self.default_parameters.items():
            if parameter_name in exp.parameters:
                vslamlab_command += [f"{str(parameter_name)}:{str(exp.parameters[parameter_name])}"]
            else:
                vslamlab_command += [f"{str(parameter_name)}:{str(parameter_value)}"]

        vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute " + ' '.join(vslamlab_command)
        return vslamlab_command

    def build_execute_command_python(self, exp_it, exp, dataset, sequence_name):
        sequence_path = os.path.join(dataset.dataset_path, sequence_name)
        exp_folder = os.path.join(exp.folder, dataset.dataset_folder, sequence_name)
        calibration_yaml = os.path.join(sequence_path, 'calibration.yaml')
        rgb_exp_txt = os.path.join(exp_folder, 'rgb_exp.txt')

        vslamlab_command = [f"--sequence_path {sequence_path}",
                            f"--calibration_yaml {calibration_yaml}",
                            f"--rgb_txt {rgb_exp_txt}",
                            f"--exp_folder {exp_folder}",
                            f"--exp_it {exp_it}",
                            f"--settings_yaml {self.settings_yaml}"]

        for parameter_name, parameter_value in self.default_parameters.items():
            if parameter_name in exp.parameters:
                vslamlab_command += [f"--{str(parameter_name)} {str(exp.parameters[parameter_name])}"]
            else:
                vslamlab_command += [f"--{str(parameter_name)} {str(parameter_value)}"]

        vslamlab_command = f"pixi run --frozen -e {self.baseline_name} execute " + ' '.join(vslamlab_command)
        return vslamlab_command
