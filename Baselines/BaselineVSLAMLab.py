import subprocess

from utilities import ws


class BaselineVSLAMLab:

    def __init__(self, baseline_name, baselines_path):
        self.baseline_name = baseline_name
        self.baselines_path = baselines_path
        self.default_parameters = ''

    def get_default_parameters(self):
        return self.default_parameters

    def execute(self, command, log_file_path):
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
