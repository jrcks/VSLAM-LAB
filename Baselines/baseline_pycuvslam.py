import os
import subprocess
from typing import Tuple, Any

from utilities import print_msg
from Baselines.BaselineVSLAMLab import BaselineVSLAMLab

SCRIPT_LABEL = f"\033[95m[{os.path.basename(__file__)}]\033[0m "


class PyCuVSLAM_baseline(BaselineVSLAMLab):
    """
    PyCuVSLAM baseline implementation for VSLAM evaluation.
    
    This class handles the setup, installation, and execution of PyCuVSLAM
    within the VSLAMLab framework.
    """
    
    def __init__(self, baseline_name: str = 'pycuvslam', baseline_folder: str = 'PyCuVSLAM'):
        default_parameters = {
            'verbose': 1, 
            'mode': 'mono',
            'visualization': False
        }
        super().__init__(baseline_name, baseline_folder, default_parameters)
        
        self.color = 'cyan'
        self.modes = ['mono']
        
        # Configure CUDA environment
        os.environ['CONDA_OVERRIDE_CUDA'] = '12.6'
        
        # Update LD_LIBRARY_PATH if needed
        conda_prefix = os.environ.get('CONDA_PREFIX')
        if conda_prefix:
            current_ld_path = os.environ.get('LD_LIBRARY_PATH', '')
            new_ld_path = os.path.join(conda_prefix, 'lib')
            
            if new_ld_path not in current_ld_path.split(':'):
                os.environ['LD_LIBRARY_PATH'] = f"{new_ld_path}:{current_ld_path}"
                print_msg(SCRIPT_LABEL, f"Updated LD_LIBRARY_PATH to include {new_ld_path}")

    def build_execute_command(self, exp_it: int, exp: Any, dataset: Any, sequence_name: str) -> str:
        return super().build_execute_command_python(exp_it, exp, dataset, sequence_name)

    def is_installed(self) -> Tuple[bool, str]:
        """
        Check if PyCuVSLAM is properly installed.
        
        Returns:
            Tuple of (is_installed: bool, status_message: str)
        """
        if not os.path.isdir(self.baseline_path):
            return False, 'Repository not cloned'
        
        try:
            # Check if the package can be imported
            result = subprocess.run([
                "pixi", "run", "-e", "pycuvslam", "python", "-c",
                "import cuvslam; print('installed')"
            ], capture_output=True, text=True, check=False, 
            cwd=self.baseline_path)
            
            if result.returncode == 0 and 'installed' in result.stdout:
                return True, 'is installed and importable'
            
            # Check if package exists but isn't importable
            pip_result = subprocess.run([
                "pixi", "run", "-e", "pycuvslam", "pip", "list"
            ], capture_output=True, text=True, check=False, timeout=30)
            
            if 'cuvslam' in pip_result.stdout.lower():
                return False, 'package found but not importable - may need reinstallation'
            else:
                return False, 'package not found or not importable'
                
        except Exception as e:
            return False, f'installation check failed: {str(e)}'
