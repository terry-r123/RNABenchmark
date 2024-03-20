# RNABenchmark

Repository for RNABenchmark

## Prerequisites

### Pretraining and Finetuning

- Python
- PyTorch
- Transformers

### Tasks and Datasets

Data can be found in path "/mnt/data/oss_beijing/multi-omics/RNA/downstream/" on 3090 oss or DATA_PATH in script file.

#### Data
- [x] Non-coding Function Classification
- [x] Mean Ribosome Loading
- [x] Vaccine Degradation Prediction
- [x] Secondary structure prediction
- [x] Modification Prediction
- [x] Contact map prediction
- [x] Distance map prediction

- [x] Splice site prediction

- [x] Isoform

- [x] Programmable RNA Switches
- [ ] CRISPR
- [ ] Prime Edting
#### Code
- [x] Non-coding Function Classification
- [x] Mean Ribosome Loading
- [x] Vaccine Degradation Prediction
- [x] Secondary structure prediction
- [ ] Modification Prediction  solving small problem
- [ ] Contact map prediction
- [ ] Distance map prediction

- [ ] Splice site prediction

- [ ] Isoform

- [ ] Programmable RNA Switches
- [ ] CRISPR
- [ ] Prime Edting


### Analysis

Results can be found in "https://aicarrier.feishu.cn/sheets/KyNGs5sWoh7tGBtQ5vkczWC6nIf?sheet=iUfeNc" on Feishu.


## Usage

run the bash scripts in the `scripts` folder
### Problems & Bugs

### Technical Details

- [x] Figure out the exact procedure of the pretraining task and the finetuning task. Figure out what does each piece of code do. Figure out the data format of the input and output of each piece of code.
- [x] Figure out how to use transformer Block to replace the Hyena Block. (Just use the config)
- [ ] Define more downstream tasks and test the performance of the model.
- [ ] Use vLLM (Paged Attention) to speed up and save memory
- [ ] Use the new version of Flash Attention
