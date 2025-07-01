# FINER: Flexible spectral-bias tuning in Implicit NEural Representation by Variable-periodic Activation Functions

## CVPR 2024

[Zhen Liu](https://liuzhen0212.github.io/)<sup>1,\*</sup>,
[Hao Zhu](https://pakfa.github.io/zhuhao_photo.github.io/)<sup>1,\*</sup>,
[Qi Zhang](https://qzhang-cv.github.io/)<sup>2</sup>,
[Jingde Fu](https://fiddiemath.github.io/)<sup>1</sup>,
[Weibing Deng](https://math.nju.edu.cn/szdw/apypl1/20190916/i22123.html)<sup>1</sup>,
[Zhan Ma](https://vision.nju.edu.cn/main.htm)<sup>1</sup>,
[Yanwen Guo](https://cs.nju.edu.cn/ywguo/index.htm)<sup>1</sup>,
[Xun Cao](https://cite.nju.edu.cn/People/Faculty/20190621/i5054.html)<sup>1</sup>,

<sup>1</sup>Nanjing University, <sup>2</sup>Tencent AI Lab, <sup>\*</sup>Equal contibution

## [Project Page](https://liuzhen0212.github.io/finer/) | [Paper](https://arxiv.org/abs/2312.02434)

<!-- We propose a novel implicit neural representation with flexible spectral-bias tuning for representing and optimizing signals. The repo contains the codes for `image fitting`, `sdf fitting & evaluation`, `ntk visualization`.

For the SDF and NeRF experiments, we utilized the codes of [Bacon](https://github.com/computational-imaging/bacon) and [torch-ngp](https://github.com/ashawkey/torch-ngp), respectively. For the neural tangent kernel (NTK) visualization, we utilized the code of [inr_dictionaries](https://github.com/gortizji/inr_dictionaries). -->
We introduce a novel implicit neural representation that allows for flexible tuning of the spectral bias, enhancing signal representation and optimization. 🚀

This repository provides the code for several applications:

* **Image Fitting:** Demonstrates the model's ability to represent 2D images.
* **SDF Fitting & Evaluation:** Includes code for fitting signed distance functions and evaluating the results, based on the [**Bacon**](https://github.com/computational-imaging/bacon) repository.
* **NeRF Implementation:** Our NeRF experiments are built upon the [**torch-ngp**](https://github.com/ashawkey/torch-ngp) codebase.
* **NTK Visualization:** We utilize code from [**inr_dictionaries**](https://github.com/gortizji/inr_dictionaries) to visualize the neural tangent kernel, offering insights into the model's behavior.


<div align=center>
<img src="img/activations.png" alt="Activations" width="70%">
</div>

## Setup
```bash
conda create -n finer python=3.8
conda activate finer
pip install -r requirements.txt
```

## Training

### Image Fitting
```bash
bash run_finer.sh 
# run_siren.sh; run_pemlp.sh; run_gauss.sh; run_wire.sh
```

### SDF Fitting & Evaluation
Setup a conda environment based on [Bacon](https://github.com/computational-imaging/bacon) and run [download_datasets.py
](https://github.com/liuzhen0212/FINER/blob/main/sdf/bacon/download_datasets.py) to download datasets.  

```bash
cd sdf/bacon/experiments
conda activate bacon

## train 
bash run_paper_finer.sh # siren, wire, guass, wire-finer, guass-finer

## evaluation
python eval.py
```

## NTK Visualization
Setup a conda environment based on [inr_dictionaries](https://github.com/gortizji/inr_dictionaries).
```bash
cd ntk
run ntk.ipynb
```


## Neurons Visualizations
- [The outputs of the first layer neurons](https://github.com/liuzhen0212/FINER/blob/main/figure_6/firstlayer_neurons.ipynb)


## Citation
```BibTeX
@inproceedings{liu2024finer,
    title = {FINER: Flexible spectral-bias tuning in Implicit NEural Representation by Variable-periodic Activation Functions},
    author = {Liu, Zhen and Zhu, Hao and Zhang, Qi and Fu, Jingde and Deng, Weibing and Ma, Zhan and Guo, Yanwen and Cao, Xun},
    booktitle = {Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition Conference (CVPR)},
    year = {2024}
}
```
