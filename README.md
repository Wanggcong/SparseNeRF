<div align="center">

<h1>SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis </h1>

<div>
    <a href='https://wanggcong.github.io/' target='_blank'>Guangcong Wang</a>&emsp;
    <a href='https://frozenburning.github.io/' target='_blank'>Zhaoxi Chen</a>&emsp;
    <a href='https://www.mmlab-ntu.com/person/ccloy/' target='_blank'>Chen Change Loy</a>&emsp;
    <a href='https://liuziwei7.github.io/' target='_blank'>Ziwei Liu</a>
</div>
<div>
    S-Lab, Nanyang Technological University
</div>

<div>
    Under review
</div>




### [Project](https://style-light.github.io/) | [YouTube](https://www.youtube.com/watch?v=sHeWK1MSPg4) | [arXiv](https://arxiv.org/abs/2207.14811) 
<tr>
    <img src="https://github.com/wanggcong/wanggcong.github.io/blob/master/projects/SparseNeRF/static/images/demo_video_v4_scenes_only_sampled3.gif" width="90%"/>
</tr>
</div>

>**Abstract:** Neural Radiance Field (NeRF) has achieved remarkable results in synthesizing photo-realistic novel views with implicit function representations. However, NeRF significantly degrades when only a few views are available. To address this problem, existing few-shot NeRF methods impose sparsity and continuity regularizations on geometry (e.g., density and depth), or exploit high-level semantics to guide the learning of NeRF. Although these methods reduce degradation in few-shot scenarios, they still struggle to synthesize photo-realistic novel views due to insufficient 3D constraints. To complement the lack of 3D information, we present a new Sparse-view NeRF (SparseNeRF) framework that effectively exploits robust depth priors from a large pre-trained depth model. Since the depth estimation of large pre-trained depth models is coarse, we propose a local depth ranking constraint on NeRF such that the expected depth ranking of the NeRF is consistent with that of the pre-trained depth model in local patches. To preserve spatial continuity of the estimated depth of NeRF, we further propose a spatial continuity constraint such that the expected depth continuity of NeRF is consistent with that of the pre-trained depth model. With the distilled depth priors of large pre-trained depth models, SparseNeRF outperforms all of the state-of-the-art few-shot NeRF methods. Extensive experiments on the LLFF and DTU benchmarks show the effectiveness and superiority of SparseNeRF. Code and models will be released.



## 1. Prerequisites
- Linux or macOS
- Python 3.6.13
- NVIDIA GPU + CUDA cuDNN(10.1)
- OpenCV

## 2. Installation
We recommend using the virtual environment (conda) to run the code easily.

```
conda create -n sparsenerf python=3.6.13
conda activate sparsenerf
pip install -r requirements.txt
```

Download jax+cuda (jaxlib-0.1.68+cuda101-cp36) wheels from [this link](https://storage.googleapis.com/jax-releases/jax_cuda_releases.html) by

```
wget https://storage.googleapis.com/jax-releases/cuda101/jaxlib-0.1.68+cuda101-cp36-none-manylinux2010_x86_64.whl
```

```
pip install jaxlib-0.1.68+cuda101-cp36-none-manylinux2010_x86_64.whl
```

```
rm jaxlib-0.1.68+cuda101-cp36-none-manylinux2010_x86_64.whl
```

Install pytorch and related packages for pretrained depth models
```
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.1 -c pytorch
pip install timm
pip install opencv-python
```

Install ffmpeg for composing videos
```
pip install imageio-ffmpeg
```
## 3. Dataset

### Download DTU dataset
- Download the DTU dataset from the [official website](https://roboimagedata.compute.dtu.dk/?page_id=36/), "Rectified (123 GB)" and "SampleSet (6.3 GB)"
- Data: extract "Rectified (123 GB)"
- Poses: extract "SampleSet/MVS\ Data/Calibration/cal18/" from "SampleSet (6.3 GB)"
- Masks: download masks (used for evaluation only) from [this link](https://drive.google.com/file/d/1Yt5T3LJ9DZDiHbtd9PDFNHqJAd7wt-_E/view?usp=sharing)


### Download LLFF dataset
- Download LLFF from [the official download link](https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1).


### Pre-process datasets
- Get depth maps
- For both LLFF and DTU, please set the variables $root_path, $benchmark, and $dataset_id in get_depth_map.sh, and run:
```
sh scripts/get_depth_map.sh

```


## 4. Training 

### 4.1 Training on LLFF
Please set the variables in scripts/train_llff.sh and configs/llff3.gin, and run:
```
sh scripts/train_llff.sh
```


### 4.2 Training on DTU
Please set the variables in train_dtu3.sh, and run:

```
sh scripts/train_dtu3.sh
```


## 5. Test 
### 5.1 Evaluate PSNR and SSIM
Please set the variables (the same as train_llff3.sh and train_dtu3.sh) in eval_llff3.sh or eval_dtu3, and run:
```
sh scripts/eval_llff3.sh
```

or 
```
sh scripts/eval_dtu3.sh
```
### 5.2 (optional) Render videos
Please set the variables (the same as train_llff3.sh and train_dtu3.sh) in render_video_llff3.sh or render_video_dtu3, and run:
```
sh scripts/render_video_llff3.sh
```

or 
```
sh scripts/render_video_dtu3.sh
```

### 5.3 (optional) Compose videos
Please set the variables in get_video.sh, and run:
```
sh get_video.sh
```

### 5.4 (optional) Tensorboard for visualizing training if necessary.
```
tensorboard --logdir=./out/xxx/ --port=6006
```
If it raises errors, see Q2 of [FQA](https://github.com/Wanggcong/SparseNeRF/blob/master/FQA.md)


## 6. To-Do
- [x] Training code
- [x] Inference model
- [x] Clean Code
- [ ] Colab Demo



## 7. Citation

If you find this useful for your research, please cite the our paper.

```
@inproceedings{wang2022sparsenerf,
   author    = {Wang, Guangcong and Chen, Zhaoxi and Loy, Chen Change and Liu, Ziwei},
   title     = {SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis},
   booktitle = {Under review},   
   year      = {2022},
  }
```

or
```
Guangcong Wang, Zhaoxi Chen, Chen Change Loy, and Ziwei Liu. SparseNeRF: Distilling Depth Ranking for Few-shot Novel View Synthesis, Under review.
```

## 8. Related Links
[RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs, CVPR, 2022](https://m-niemeyer.github.io/regnerf/index.html)

[StyleLight: HDR Panorama Generation for Lighting Estimation and Editing, ECCV 2022](https://github.com/Wanggcong/StyleLight).

[Relighting4D: Neural Relightable Human from Videos, ECCV 2022](https://github.com/FrozenBurning/Relighting4D)




## 9. Acknowledgments
This code is based on the [RegNeRF](https://github.com/google-research/google-research/tree/master/regnerf) and [DPT](https://github.com/isl-org/DPT) codebases. 

## 10. FAQ
We will summarize frequently asked questions at this link [FAQ](https://github.com/Wanggcong/SparseNeRF/blob/master/FQA.md).
