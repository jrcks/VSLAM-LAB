import argparse, os, yaml, time
from inputimeout import inputimeout, TimeoutOccurred
import pandas as pd

from Run.run_functions import run_sequence
from vslamlab_eval import evaluate, compare
from Datasets.get_dataset import get_dataset
from utilities import ws, show_time, filter_inputs
from Baselines.baseline_utilities import get_baseline
from vslamlab_utilities import load_experiments, check_config_integrity
from path_constants import VSLAMLAB_BENCHMARK, VSLAMLAB_EVALUATION, EXP_YAML_DEFAULT

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "

def main():

    parser = argparse.ArgumentParser(description=f"{__file__}")

    parser.add_argument('--exp_yaml', nargs='?', type=str,
                        const=EXP_YAML_DEFAULT, default=EXP_YAML_DEFAULT,
                        help=f"Path to the YAML file containing the list of experiments. "
                             f"Default \'vslamlab --exp_yaml {EXP_YAML_DEFAULT}\'")

    parser.add_argument('-download', action='store_true', help="")
    parser.add_argument('-run', action='store_true', help="")
    parser.add_argument('-evaluate', action='store_true', help="")
    parser.add_argument('-compare', action='store_true', help="")

    parser.add_argument('-ablation', action='store_true', help="")
    parser.add_argument('-overwrite', action='store_true', help="")

    args = parser.parse_args()

    if not os.path.exists(VSLAMLAB_EVALUATION):
        os.makedirs(VSLAMLAB_EVALUATION, exist_ok=True)

    # Load experiment info
    experiments, config_files = load_experiments(args.exp_yaml, overwrite=args.overwrite)
    check_config_integrity(config_files)

    # Process experiments
    filter_inputs(args)
    if args.download:
        download(config_files)

    if args.run:
        run(experiments, args.exp_yaml, ablation=args.ablation)

    if args.evaluate:
        evaluate(experiments, overwrite=args.overwrite)

    if args.compare:
        compare(experiments, args.exp_yaml)

def run(experiments, exp_yaml, ablation=False):
    print(f"\n{SCRIPT_LABEL}Running experiments (in {exp_yaml}) ...")
    start_time = time.time()

    completed_runs = {}
    not_completed_runs = {}
    num_executed_runs = 0
    duration_time_total = 0

    all_experiments_completed = False
    while not all_experiments_completed:
        remaining_iterations = 0
        for [exp_name, exp] in experiments.items():
            exp_log = pd.read_csv(exp.log_csv)
            completed_runs[exp_name] = (exp_log["STATUS"] == "completed").sum()  
            not_completed_runs[exp_name] = (exp_log["STATUS"] != "completed").sum() 
            remaining_iterations += not_completed_runs[exp_name]

            if not_completed_runs[exp_name] == 0:
                continue
                
            first_not_finished_experiment = exp_log[exp_log["STATUS"] != "completed"].index.min()
            row = exp_log.loc[first_not_finished_experiment]
            baseline = get_baseline(row['method_name'])
            dataset = get_dataset(row['dataset_name'], VSLAMLAB_BENCHMARK)    

            results = run_sequence(row['exp_it'], exp, baseline, dataset, row['sequence_name'], ablation)

            duration_time = results['duration_time']
            duration_time_total += duration_time
            num_executed_runs += 1
            remaining_iterations -= 1

            exp_log["STATUS"] = exp_log["STATUS"].astype(str)
            exp_log["SUCCESS"] = exp_log["SUCCESS"].astype(str)
            exp_log["COMMENTS"] = exp_log["COMMENTS"].astype(str)
            exp_log.loc[first_not_finished_experiment, "STATUS"] = "completed"
            exp_log.loc[first_not_finished_experiment, "SUCCESS"] = results['success']
            exp_log.loc[first_not_finished_experiment, "COMMENTS"] = results['comments']
            exp_log.loc[first_not_finished_experiment, "TIME"] = duration_time
            exp_log.to_csv(exp.log_csv, index=False)
                
        all_experiments_completed = exp_log['STATUS'].eq("completed").all()
        if(duration_time_total > 1):
            print(f"\n{SCRIPT_LABEL}: Experiment report: {exp_yaml}")
            print(f"{ws(4)}\033[93mNumber of executed iterations: {num_executed_runs} / {num_executed_runs + remaining_iterations} \033[0m")
            print(f"{ws(4)}\033[93mAverage time per iteration: {show_time(duration_time_total / num_executed_runs)}\033[0m")
            print(f"{ws(4)}\033[93mTotal time consumed: {show_time(duration_time_total)}\033[0m")
            print(f"{ws(4)}\033[93mRemaining time until completion: {show_time(remaining_iterations * duration_time_total / num_executed_runs)}\033[0m")

    run_time = (time.time() - start_time)
    print(f"\033[93m[Experiment runtime: {show_time(run_time)}]\033[0m")

def download(config_files):
    download_issues = find_download_issues(config_files)
    solve_download_issues(download_issues)

    print(f"\n{SCRIPT_LABEL}Downloading (to {VSLAMLAB_BENCHMARK}) ...")

    for config_file in config_files:
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    dataset.download_sequence(sequence_name)

def estimate_experiments_time(experiments):
    running_time = 0
    for [exp_name, exp] in experiments.items():
        with open(exp.config_yaml, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                for sequence_name in sequence_names:
                    num_frames = dataset.get_sequence_num_rgb(sequence_name)
                    running_time += exp.num_runs * num_frames / dataset.rgb_hz
    return 1.5 * running_time / 60


def find_download_issues(config_files):
    download_issues = {}
    for config_file in config_files:
        with open(config_file, 'r') as file:
            config_file_data = yaml.safe_load(file)
            for dataset_name, sequence_names in config_file_data.items():
                dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
                download_issues[dataset.dataset_name] = {}
                for sequence_name in sequence_names:
                    sequence_availabilty = dataset.check_sequence_availability(sequence_name)
                    if sequence_availabilty != "available":
                        issues_seq = dataset.get_download_issues(sequence_name)
                        if issues_seq == {}:
                            continue
                        for issue_name, issue_topic in issues_seq.items():
                            download_issues[dataset.dataset_name][issue_name] = issue_topic

    print(f"\n{SCRIPT_LABEL}Finding download issues...")
    num_download_issues = 0
    for dataset_name, issues_dataset in download_issues.items():
        for issue_name, issue_topic in issues_dataset.items():
            print(f"{ws(4)}[{dataset_name}][{issue_name}]: {issue_topic}")
            num_download_issues += 1

    if num_download_issues > 0:
        message = (f"\n{SCRIPT_LABEL}Found download issues: your experiments have {num_download_issues} download "
                   f"issues. Would you like to continue solving them and download the datasets (Y/n):")
        try:
            user_input = inputimeout(prompt=message, timeout=120).strip().upper()
        except TimeoutOccurred:
            user_input = 'Y'
            print("        No input detected. Defaulting to 'Y'.")
        if user_input != 'Y':
            exit()
    else:
        message = (f"{ws(4)}Found download issues: your experiments have {num_download_issues} download "
                   f"issues.")
        print(message)
        download_issues = {}

    return download_issues


def solve_download_issues(download_issues):
    if download_issues == {}:
        return

    print(f"\n{SCRIPT_LABEL}Solving download issues: ")
    for dataset_name, issues_dataset in download_issues.items():
        dataset = get_dataset(dataset_name, VSLAMLAB_BENCHMARK)
        for download_issue in issues_dataset.items():
            dataset.solve_download_issue(download_issue)


if __name__ == "__main__":
    main()
