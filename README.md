# BEACON: Benchmark for Comprehensive RNA Tasks and Language Models

This is the official codebase of the paper [BEACON: Benchmark for Comprehensive RNA Tasks and Language Models](https://arxiv.org/abs/2406.10391)



## 🔥 Update
- [06/11]🔥BEACON is coming! We release the [paper](https://arxiv.org/abs/2406.10391), [code](https://github.com/terry-r123/RNABenchmark), [data](https://drive.google.com/drive/folders/19ddrwI8ycvIxkgSV3gDo_VunLofYd4-6?usp=sharing), and [models](https://drive.google.com/drive/folders/1455JIOGV5X96CCgxCT-QgVu0xbXFz72X?usp=sharing) for BEACON!

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

## 🔍 Tasks and Datasets

Datasets of RNA tasks can be found in [Google Drive](https://drive.google.com/drive/folders/19ddrwI8ycvIxkgSV3gDo_VunLofYd4-6?usp=sharing)

Model checkpoints of opensource RNA LM and BEACON-B can be found in [Google Drive](https://drive.google.com/drive/folders/1455JIOGV5X96CCgxCT-QgVu0xbXFz72X?usp=sharing)



## 🔍 Usage
To evalute on all RNA tasks, you can run the bash scripts in the `scripts` folder, for example:
```
cd RNABenchmark
bash ./scripts/BEACON-B/all_task.sh
```
## License ##

This codebase is released under the Apache License 2.0 as in the [LICENSE](LICENSE) file.

## Citation
If you find this repo useful for your research, please consider citing the paper
```
@misc{ren2024beacon,
      title={BEACON: Benchmark for Comprehensive RNA Tasks and Language Models}, 
      author={Yuchen Ren and Zhiyuan Chen and Lifeng Qiao and Hongtai Jing and Yuchen Cai and Sheng Xu and Peng Ye and Xinzhu Ma and Siqi Sun and Hongliang Yan and Dong Yuan and Wanli Ouyang and Xihui Liu},
      year={2024},
      eprint={2406.10391},
      archivePrefix={arXiv},
      primaryClass={id='q-bio.QM' full_name='Quantitative Methods' is_active=True alt_name=None in_archive='q-bio' is_general=False description='All experimental, numerical, statistical and mathematical contributions of value to biology'}
}
```

