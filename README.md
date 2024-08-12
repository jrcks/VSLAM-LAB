<p align="center">

  <h1 align="center"> VSLAM-LAB 
  <h3 align="center"> A Comprehensive Framework for Visual SLAM Systems and Datasets</h3> 
  </h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=SDtnGogAAAAJ&hl=en"><strong>Alejandro Fontan</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=j_sMzokAAAAJ&hl=en"><strong>Javier Civera</strong></a>
    ·
    <a href="https://scholar.google.com/citations?user=TDSmCKgAAAAJ&hl=en"><strong>Michael Milford</strong></a>
  </p>


Compile and configure **Visual SLAM** systems, download and process the **datasets**, design and run your own experiments, and evaluate and analyze the results, **all with a single command line!**
```
pixi run vslamlab --exp_yaml configs/experiments.yaml
```
**VSLAM-LAB** aims to unify and simplify the development, evaluation, and application of VSLAM systems. It provides a comprehensive framework and tools to seamlessly download, compile, and configure VSLAM systems and datasets. **VSLAM-LAB** allows users to easily set up experiments and evaluate and compare results, making VSLAM more accessible and broadly useful.
 
# Getting Started

To ensure all dependencies are installed in a reproducible manner, we use the package management tool [**pixi**](https://pixi.sh/latest/). If you haven't installed [**pixi**](https://pixi.sh/latest/) yet, please run the following command in your terminal:
```
curl -fsSL https://pixi.sh/install.sh | bash
```
You might need to restart your terminal or source your shell for the changes to take effect. For more details, refer to the [**pixi documentation**](https://pixi.sh/latest/).


## Installation

Clone the repository and navigate to the project directory:
```
git clone https://github.com/alejandrofontan/VSLAM-LAB.git && cd VSLAM-LAB
```
Next, **download** and **build** the following V-SLAM systems: [**AnyFeature-VSLAM**](https://github.com/alejandrofontan/AnyFeature-VSLAM), [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2) and [**DSO**](https://github.com/alejandrofontan/dso). You can build all components at once using the command:
```
pixi run build-all
```
*Alternatively, you can build each system separately with the following commands:*
```
pixi run -e anyfeature build
pixi run -e orbslam2 build
pixi run -e dso build
```
*To change the paths where VSLAM-LAB-Benchmark or/and VSLAM-LAB-Evaluation data are stored (for example, to /media/${USER}/data), use the following commands:*
```
pixi run set-benchmark-path /media/${USER}/data
pixi run set-evaluation-path /media/${USER}/data
```
To **run** the demo, use the following command:

```
pixi run vslamlab --exp_yaml configs/exp_demo.yaml
```

**Note:** The demo will execute one run per sequence using each VSLAM system. There are 80 pre-executed runs saved in VSLAM-LAB-Evaluation to assist with visualization purposes. The demo uses modified versions of [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2) and [**DSO**](https://github.com/alejandrofontan/dso). Please note that the comparison is between SLAM and Odometry and is intended only as an example of how to use **VSLAM-LAB**.

# Configure your own experiment
Experiments in **VSLAM-LAB** are defined as a sequence of entries in a YAML file (see example **~/VSLAM-LAB/configs/exp_demo_short.yaml**):
```
exp_01:
  VSLAM: dso                      # VSLAM system (anyfeature/orbslam2/dso)
  Parameters: {}                  # Vector with parameters that will be input to the VSLAM executable 
  Config: config_demo_short.yaml  # YAML file containing the sequences to be run 
  NumRuns: 1                      # Maximum number of executions per sequence
```

**Config** files are YAML files containing the list of sequences to be executed in the experiment (see example **~/VSLAM-LAB/configs/config_demo_short.yaml**):
```
rgbdtum: 
  - 'rgbd_dataset_freiburg1_xyz'
eth: 
  - 'table_3'
  - 'planar_2'
kitti:
  - '04'
  - '09'
euroc:
  - 'MH_01_easy'
monotum:
  - 'sequence_01'
```

For a list of both available VSLAM systems and the available collections of datasets, please check the section [VSLAM-LAB Leaderboard](#vslam-lab-leaderboard).

# Add a new dataset

Datasets in **VSLAM-LAB** are stored in a folder named **VSLAM-LAB-Benchmark**, which is created by default in the same parent directory as **VSLAM-LAB**. If you want to modify the location of your datasets, change the variable **VSLAMLAB_BENCHMARK** in **~/VSLAM-LAB/utilities.py**.

1. To add a new dataset, structure your dataset as follows:
```
~/VSLAM-LAB-Benchmark
└── YOUR_DATASET
    └── sequence_01
        ├── rgb
            └── img_01
            └── img_02
            └── ...
        ├── calibration.yaml
        ├── rgb.txt
        └── groundtruth
    └── sequence_02
        ├── ...
    └── ...   
```

2. Derive a new class **dataset_{your_dataset}.py** for your dataset from  **~/VSLAM-LAB/Datasets/Dataset_vslamlab.py**, and create a corresponding YAML configuration file named **dataset_{your_dataset}.yaml**.
	
3. Include the call for your dataset in function *def get_dataset(...)* in **~/VSLAM-LAB/Datasets/Dataset_utilities.py**
```
 from Datasets.dataset_{your_dataset} import {YOUR_DATASET}_dataset
    ...
 def get_dataset(dataset_name, benchmark_path)
    ...
    switcher = {
        "rgbdtum": lambda: RGBDTUM_dataset(benchmark_path),
        ...
        "dataset_{your_dataset}": lambda: {YOUR_DATASET}_dataset(benchmark_path),
    }
    
```

# Add a new VSLAM baseline

# License
**VSLAM-LAB** is released under a **LICENSE.txt**. For a list of code dependencies which are not property of the authors of **VSLAM-LAB**, please check **docs/Dependencies.md**.


# Citation
If you're using **VSLAM-LAB** in your research, please cite. If you're specifically using VSLAM systems or datasets that have been included, please cite those as well. We provide a [spreadsheet](https://docs.google.com/spreadsheets/d/1V8_TLqlccipJ6x_TXkgLsw9zWszHU9M-0mGgDT92TEs/edit?usp=drive_link) with citation for each dataset and VSLAM system for your convenience. 

# Acknowledgements

To [awesome-slam-datasets](https://github.com/youngguncho/awesome-slam-datasets)
# VSLAM-LAB Leaderboard

| VSLAM                                                                                                               | Parameters    | License | Label         |
|:--------------------------------------------------------------------------------------------------------------------|:-------------:|:-------:|:-------------:|
| [**AnyFeature-VSLAM**](https://github.com/alejandrofontan/AnyFeature-VSLAM)                                         |  --           | ---- -- | `anyfeature`  |
| [**DSO**](https://github.com/alejandrofontan/dso)                                                                   |  --           | ---- -- | `dso`         |
| [**ORB-SLAM2**](https://github.com/alejandrofontan/ORB_SLAM2)                                                       |  --           | ---- -- | `orbslam2`    | 


| Dataset                                                                                                             | Seq    | Size    | Label         |
|:--------------------------------------------------------------------------------------------------------------------|:------:|:-------:|:-------------:|
| [**ETH3D SLAM Benchmarks**](https://www.eth3d.net/slam_datasets)                                                    |  87    | 33.7 GB | `eth`         |
| [**RGB-D SLAM Dataset and Benchmark**](https://cvg.cit.tum.de/data/datasets/rgbd-dataset)                           |  80    | 61.6 GB | `rgbdtum`     |
| [**ICL-NUIM RGB-D Benchmark Dataset**](https://www.doc.ic.ac.uk/~ahanda/VaFRIC/iclnuim.html)                        |  16    |  6.4 GB | `nuim`        | 
| [**Monocular Visual Odometry Dataset**](https://cvg.cit.tum.de/data/datasets/mono-dataset)                          |  50    | 15.4 GB | `monotum`     |
| [**The KITTI Vision Benchmark Suite**](https://www.cvlibs.net/datasets/kitti/eval_odometry.php)                     |  22    | 11.5 GB | `kitti`       |
| [**RGB-D Dataset 7-Scenes**](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)              |  46    | 17.9 GB | `7scenes`     |
| [**The EuRoC MAV Dataset**](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)           |  ?     |  ? GB   | `euroc`       |
| [**TartanAir: A Dataset to Push the Limits of Visual SLAM**](https://theairlab.org/tartanair-dataset/)              |  16    | 24.6 GB | `tartanair`   |
