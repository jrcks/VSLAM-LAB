import sys, os, yaml, shutil, csv
from colorama import Fore, Style

from Datasets.get_dataset import get_dataset
from Baselines.baseline_utilities import get_baseline
from utilities import ws, check_yaml_file_integrity
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION, VSLAM_LAB_DIR, CONFIG_DEFAULT

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

class Experiment:
    def __init__(self, name, exp_folder, num_runs, method, parameters, config_yaml=CONFIG_DEFAULT, ablation_csv=None, overwrite=False):
        self.name = name
        self.folder = exp_folder
        self.num_runs = num_runs
        self.module = method
        self.parameters = parameters

        self.log_csv = os.path.join(self.folder, 'vslamlab_exp_log.csv')
        self.config_yaml = os.path.join(VSLAM_LAB_DIR, 'configs', config_yaml)
        self.ablation_csv = ablation_csv


        if os.path.exists(self.folder):
            if overwrite:
                shutil.rmtree(self.folder)
                print(f"{SCRIPT_LABEL}" + Fore.YELLOW + f"Warning: The folder '{self.folder}' already exists. Overwriting it." + Style.RESET_ALL)
            else:
                print(f"{SCRIPT_LABEL}" + Fore.YELLOW + f"Warning: The folder '{self.folder}' already exists. Continue execution." + Style.RESET_ALL)
            
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)
        
        if not os.path.exists(self.log_csv):
            log_headers = ["method_name", "dataset_name", "sequence_name", "exp_it", "STATUS", "SUCCESS", "TIME", "MEMORY", "COMMENTS", "EVALUATION"]

            with open(self.log_csv, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(log_headers)
                with open(self.config_yaml, 'r') as file:
                    config_file_data = yaml.safe_load(file)
                    for i in range(num_runs):
                        for dataset_name, sequence_names in config_file_data.items():
                            for sequence_name in sequence_names:
                                exp_it = str(i).zfill(5)  
                                writer.writerow([self.module, dataset_name, sequence_name, f"{exp_it}", "", "",0.0, 0.0, "", "none"])
        

def list_datasets():
    dataset_scripts_path = os.path.join(VSLAM_LAB_DIR, 'Datasets')
    dataset_scripts = []
    for filename in os.listdir(dataset_scripts_path):
        if 'dataset_' in filename and filename.endswith('.yaml') and 'utilities' not in filename:
            dataset_scripts.append(filename)

    dataset_scripts = [item.replace('dataset_', '').replace('.yaml', '') for item in dataset_scripts]

    return dataset_scripts

def list_baselines():
    baseline_scripts_path = os.path.join(VSLAM_LAB_DIR, 'Baselines')
    baseline_scripts = []
    for filename in os.listdir(baseline_scripts_path):
        if 'baseline_' in filename and filename.endswith('.py') and 'utilities' not in filename:
            baseline_scripts.append(filename)

    baseline_scripts = [item.replace('baseline_', '').replace('.py', '') for item in baseline_scripts]

    return baseline_scripts

def baseline_info(baseline_name):
    baseline = get_baseline(baseline_name)
    baseline.info_print()

def print_datasets():
    dataset_list = list_datasets()
    print(f"\n{SCRIPT_LABEL}Accessible datasets in VSLAM-LAB:")
    for dataset in dataset_list:
        print(f" - {dataset}")
    print("")

def print_baselines():
    baseline_list = list_baselines()
    print(f"\n{SCRIPT_LABEL}Accessible baselines in VSLAM-LAB:")
    for baseline in baseline_list:
        print(f" - {baseline}")
    print("For detailed information about a baseline, use 'pixi run baseline-info <baseline_name>'")


def load_experiments(exp_yaml, overwrite):
    """
    Loads experiment configurations from a YAML file and initializes Experiment objects.

    Parameters
    ----------
    exp_yaml : str
        Path to the YAML file containing experiment settings (default: VSLAM-LAB/docs/experimentList.yaml).

    Returns
    ----------
    experiments : dict
        experiments<exp_name,Experiment()>
    config_files : dict
        config_files<config_yaml,False>
    """

    check_yaml_file_integrity(exp_yaml)

    with open(exp_yaml, 'r') as file:
        experiment_data = yaml.safe_load(file)

    experiments = {}
    config_files = {}
    for exp_name, settings in experiment_data.items():
        experiment = Experiment(exp_name, 
                                os.path.join(VSLAMLAB_EVALUATION, exp_name), 
                                settings.get('NumRuns', 1),
                                settings.get('Module', "default"), 
                                settings['Parameters'], 
                                os.path.join(VSLAM_LAB_DIR, 'configs', settings.get('Config', CONFIG_DEFAULT)), 
                                settings.get('Ablation', None),
                                overwrite)

        experiments[exp_name] = experiment
        config_files[experiments[exp_name].config_yaml] = False
        
    print(f"\n{SCRIPT_LABEL}Experiment summary: {os.path.basename(exp_yaml)}")
    print(f"{ws(4)} Number of experiments: {len(experiments)}")
    
    return experiments, config_files

def check_config_integrity(config_files):
    dataset_list = list_datasets()
    for config_file in config_files:
        check_yaml_file_integrity(config_file)
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                if not (dataset_name in dataset_list):
                    print(f"\n{SCRIPT_LABEL}Error in : {config_file}")
                    print(f"{ws(4)}'{dataset_name}' dataset doesn't exist")
                    print_datasets()
                    sys.exit(1)

                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    if not dataset.contains_sequence(sequence_name):
                        print(f"\n{SCRIPT_LABEL}Error in : {config_file}")
                        print(f"{ws(4)}'{dataset_name}' dataset doesn't contain sequence '{sequence_name}'")
                        print(f"\nAvailable sequences are: {dataset.sequence_names}")
                        print(f"")
                        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        function_name = sys.argv[1]
        if function_name == "baseline_info":
            baseline_info(sys.argv[2])
        if function_name == "print_datasets":
            print_datasets()
        if function_name == "print_baselines":
            print_baselines()
            