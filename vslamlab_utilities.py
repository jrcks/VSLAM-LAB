import sys
from Baselines.baseline_utilities import get_baseline

def baseline_info(baseline_name):
    baseline = get_baseline(baseline_name)
    baseline.info_print()

if __name__ == "__main__":

    if len(sys.argv) > 2:
        function_name = sys.argv[1]

        if function_name == "baseline_info":
            baseline_info(sys.argv[2])
