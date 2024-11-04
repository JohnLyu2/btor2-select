This is the repo for the Google Summer of Code 2024 project [Adaptive Algorithm Selection for Btor2 Verification Tasks](https://summerofcode.withgoogle.com/programs/2024/projects/FGmF8gS3). Btor2-Select is an algorithm selector for the word-level hardware model-checking problem described in the Btor2 language. In addtion to traditional algorithm selection approaches such as Empircal Hardness Model and Pairwise Classifier, Btor2-Selector also implements an RL-based adapative algorithm selection framework! See our GSoC'24 report for more details. 

## Requirements and Installation

#### Python
We tested our program with Python 3.12.4. You can install the required Python dependencies using `requirement.txt`:
```bash
pip install -r requirements.txt
```

#### Compile the `counts` binary (counting Btor2 features)
```bash
cd btor2feature
./configure.sh
cd build
make
```

## Component Verifier Performance Data

The performance data of each component verifier are stored in `performance_data/performance.table.csv`. They were collected from verifier-instance executions on Ubuntu 22.04 machines, each with a 3.4 GHz CPU (Intel Xeon E3-1230 v5) with 8 processing units and 33 GB of RAM. Each task was assigned 2 CPU cores, 15 GB RAM, and 15 min of CPU time limit. We used [BenchExec](https://github.com/sosy-lab/benchexec) to ensure reliable resource measurement and reproducible results.  

## GSoC Results
We provide a Jupyter Notebook `reproduce_gsoc.ipynb` to interactively reproduce the GSoC results.

## AAAI'25 Student Abstract
Some preliminary results from this project will be presented at the Student Program at AAAI'25! We provide a Jupyter Notebook `reproduce_aaai.ipynb` to interactively reproduce our results in our AAAI'25 paper. 

## HWMCC'24 Submission
We submitted a sequential compositional verifier `Btor2-SelectMC` to [HWMCC'24](https://hwmcc.github.io/2024/) based on this work. Check our submission at this [Zenodo link](https://zenodo.org/records/13627812).

## License
Btor2-Select is licensed under the [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0). The submodule `counts` is largely based on codes from [Btor2Tools](https://github.com/Boolector/btor2tools), which is licensed under the [MIT License](btor2kwcount/LICENSE.txt).