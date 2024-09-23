import os

VSLAM_LAB_DIR = os.path.dirname(os.path.abspath(__file__))
VSLAM_LAB_PATH = os.path.dirname(VSLAM_LAB_DIR)

VSLAMLAB_BENCHMARK = os.path.join(VSLAM_LAB_PATH, 'VSLAM-LAB-Benchmark')
VSLAMLAB_EVALUATION = os.path.join(VSLAM_LAB_PATH, 'VSLAM-LAB-Evaluation')
VSLAMLAB_BASELINES = os.path.join(VSLAM_LAB_DIR, 'Baselines')

COMPARISONS_YAML_DEFAULT = os.path.join(VSLAM_LAB_DIR, 'configs', 'comp_complete.yaml')
EXP_YAML_DEFAULT = os.path.join(VSLAM_LAB_DIR, 'configs', 'exp_demo.yaml')
CONFIG_DEFAULT = 'config_demo.yaml'

VSLAM_LAB_EVALUATION_FOLDER = 'vslamlab_evaluation'
RGB_BASE_FOLDER = 'rgb'

ABLATION_PARAMETERS_CSV = 'log_ablation_parameters.csv'