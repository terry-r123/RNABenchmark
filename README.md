# BEACON: Benchmark for Comprehensive RNA Tasks and Language Models

This is the official codebase of the paper [BEACON: Benchmark for Comprehensive RNA Tasks and Language Models]()

## Prerequisites

### Installation
import lib:
torch==1.13.1+cu117
transformers==4.38.1


```bash
git clone https://github.com/terry-r123/RNABenchmark.git
cd RNABenchmark
conda activate -n beacon python=3.8
pip install -r requirements.txt
```

### Tasks and Datasets

Data can be found in google drive



## Usage
To evalute on all RNA tasks, you can run the bash scripts in the `scripts` folder, for example:
```
cd RNABenchmark
bash ./scripts/BEACON-B/all_task.sh
```

