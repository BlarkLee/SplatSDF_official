# SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion
This is the official implementation of **SplatSDF**.
### [Project page](https://blarklee.github.io/splatsdf/) | [Paper](https://arxiv.org/abs/2411.15468/) | Primary Contact: [Runfa](https://blarklee.github.io/)

## Installation
### Requirements
Our codes requires the following environment:

- Linux (20.04, tested on Ubuntu 20.04)
- Python (>=3.8, tested on 3.8)
- PyTorch (>=2.0.0, tested on 2.0.0)
- CUDA (>=11.8, tested on 11.8)

### Install

```
git clone https://github.com/BlarkLee/splatsdf_temporary
```

Please refer to `splatsdf.yaml` and `requirements.txt` to install all required packages. We recommend using conda virtural env. Here are the additional mandatory libraries you need to install independently:
- Install pycuda following: https://documen.tician.de/pycuda/install.html
- Install torch_scatter following: https://github.com/rusty1s/pytorch_scatter
- Install Open3D following: https://www.open3d.org/docs/release/getting_started.html

## Data Preparation
### NeRF Synthetic Dataset
Please download NeRF Synthetic Dataset [here](https://www.kaggle.com/datasets/nguyenhung1903/nerf-synthetic-dataset/data).

### DTU Dataset


## Training
We get point cloud from [MVSNet](https://github.com/YoYo000/MVSNet) as [PointNeRF](). Using the point cloud, we train standard 3DGS offline from the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), the only difference is that we fix the number and coordinates of gaussians during training to keep the density of 3DGS. For convenience, we upload our trained 3DGS to [here](https://drive.google.com/drive/folders/1jCh1XGQCwdM6ogVjagz44gXwf9SWOn6d?usp=drive_link). Taking "Lego" as an example:

```
EXPERIMENT=lego
CONFIG=project/splatsdf/configs/custom/${EXPERIMENT}.yaml
python train.py --logdir=logs/${EXPERIMENT} --config=${CONFIG} --show_pbar --single_gpu 
```
Please set the correct path for data in `${CONFIG}`:
- `data.root`: path to Lego data. Eg. data/nerf_synthetic/lego
- `data.gaussian_ply_path`: path to the pretrained gaussian. Eg. `pretrained_gaussian/lego/point_cloud.ply`

And set the different ablation conditions in `${CONFIG}`:
- `model.surface_fusion`: `True`: surface fusion at the anchor point. `False`: fusion on 5 points (anchor point + 4 closest points).
- `model.gs_render_depth`: `True`: use GS-rendered depth to find the anchor points. `False`: use GS center to find the anchor points.
- `pt_rendered_depth_path`: You need to set this if you set `model.gs_render_depth` to `False`. If no depth before, the depth will be rendered by Open3D and saved at this path during the first epoch, which will be slow but the speed will be back to normal from 2nd epoch. Notice that no depth saved when using GS-rendered depth since it is fast.

Currently we support single GPU training. Our default training settings take ~23GB GPU memory. To reduce the memory requirements you can:
- Changing the number of rays per image, currently we are using 512 rays.
- Changing the number of query points per ray, currently we are using 128 points.
- Downsample 3DGS.
These options may affect the training quality.

## Inference
To extract the surface mesh with a trained model,

```
CHECKPOINT=logs/${EXPERIMENT}/epoch_xxx_iteration_xxx_checkpoint.pt
OUTPUT_MESH=logs/${EXPERIMENT}/epoch_xxx_iteration_xxx_checkpoint.ply
RESOLUTION=512
BLOCK_RES=64
python project/splatsdf/scripts/extract_mesh_surface.py --config=${CONFIG} --checkpoint=${CHECKPOINT} --output_file=${OUTPUT_MESH} --resolution=${RESOLUTION} --block_res=${BLOCK_RES} --textured --single_gpu
```

Or using our checkpoints [here](https://drive.google.com/drive/folders/1mG1B1LrfcEr5y25gsOxzH_GfjdTbDWL7?usp=drive_link) on Neuralangelo, and on our SplatSDF [here](https://drive.google.com/drive/folders/15kYpPdhQ6C9JBx-iKeorRNI6tMfQfl6Z?usp=drive_link). We only upload checkpoints for NeRF Synthetic dataset due to the drive limitation. Please email the first author if you need the checkpoint on DTU dataset.

Different to Neuralangelo using a resolution of 1024<sup>3</sup>, we use a resolution of 512<sup>3</sup> for all the results in our paper for a fast inference. Notice that our work has no influence on inference speed, choosing a low resolution will sacrifice a bit of accuracy but extremely shorten the inference time, please feel free to try high resolution.


## Evaluation
The estimated mesh need to be further cropped due to two reasons:
- The unseen region from RGB images cannot be correctly inferred.
- The ground truth point cloud of DTU dataset is bad.

For detailed explainations of cropping, please refer to the issues of Neuralangelo's repository, such as [here](https://github.com/NVlabs/neuralangelo/issues/93) and [here](https://github.com/NVlabs/neuralangelo/issues/74). 
### NeRF Synthetic Dataset
We use the mesh ground truth from [here](https://drive.google.com/drive/folders/1y7RAqQTWmW3x6xvroFiTGz4oH13jQhRF?usp=drive_link), where we randomly sample point cloud and the point cloud can be downloaded from [here](https://drive.google.com/drive/folders/1hToYPAsLYiM0plqpvx8VMLKSklj6YDjg?usp=drive_link). Download the ground truth point cloud `.ply` file and put it in `${GT_PCD}`. Download our cropped estimated mesh from [here]() 

python eval.py --data nerf --est_mesh ${OUTPUT_MESH} --gt_pcd ${GT_PCD}

### DTU Dataset
Our cropped DTU mesh estimation is in [here](https://drive.google.com/drive/folders/1CYs4GDqQlaUxprToefbqu3JeFViTrP7H?usp=drive_link). For DTU evaluation, please set the directory of `dtu_eval` to `--gt_pcd`. The scene ID need to be specified, for example scene 24:

python eval.py --data dtu --scene 24 --est_mesh ${OUTPUT_MESH} --gt_pcd xxx/DTU/dtu_eval/


## Acknowledgement
We use 3DGS code from the original [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), the SDF & color Network from [Neuralangelo](https://github.com/NVlabs/neuralangelo), and the point-based sparse KNN from [PointNeRF](https://github.com/Xharlie/pointnerf). Thanks for the contribution of these works. 

## Citation
If you find our work useful for your research, please cite
```
@misc{splatsdf,
      title={SplatSDF: Boosting Neural Implicit SDF via Gaussian Splatting Fusion}, 
      author={Runfa Blark Li and Keito Suzuki and Bang Du and Ki Myung Brian Le and Nikolay Atanasov and Truong Nguyen},
      year={2024},
      eprint={2411.15468},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.15468}, 
}
```
