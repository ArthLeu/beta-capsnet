

## 3D Point Capsule Networks
Created by <a href="http://campar.in.tum.de/Main/YongHengZhao" target="_blank">Yongheng Zhao</a>, <a href="http://campar.in.tum.de/Main/TolgaBirdal" target="_blank">Tolga Birdal</a>, <a href="http://campar.in.tum.de/Main/HaowenDeng" target="_blank">Haowen Deng</a>, <a href="http://campar.in.tum.de/Main/FedericoTombari" target="_blank">Federico Tombari </a> from TUM.

#### This is an edited version for Liu's project

See <a href="https://github.com/yongheng1991/3D-point-capsule-networks" target="_blank">this link</a> for original README documentation

Custom functions: 
1. generate capsule dataset for transfer learning, 
2. train beta-vae with capsules, 
3. decode and visualize capsules using default capsnet checkpoint

#### Chamfer Distance Package
Since the default CD package is extremely buggy, we switched to a new CD package provided by chrdiller.
Link: https://github.com/chrdiller/pyTorchChamferDistance


### Installation

The code is based on PyTorch. It has been tested with Python 3.8, PyTorch 1.6.0, CUDA 11.0(or higher) on Ubuntu 20.04.


Install h5py for Python:
```bash
  sudo apt-get install libhdf5-dev
  sudo pip install h5py
```

Install Chamfer Distance(CD) package:
```bash
  cd reference_models/pcl_models/torch_nndistance
  python3 build.py install
```


To visualize the training process in PyTorch, consider installing  <a href="https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard" target="_blank">TensorBoard</a>.

If you have GUI enabled, to visualize the reconstructed point cloud, consider installing <a href="http://www.open3d.org/docs/getting_started.html" target="_blank">Open3D</a>.
```bash
  pip3 install open3d
```

### Datasets

#### ShapeNetPart Dataset
```bash
  cd dataset
  bash download_shapenet_part16_catagories.sh
```
#### ShapeNet Core with 13 categories (refered from <a href="https://github.com/ThibaultGROUEIX/AtlasNet" target="_blank">AtlasNet</a>.)
```bash
  cd dataset
  bash download_shapenet_core13_catagories.sh
```
#### ShapeNet Core with 55 categories (refered from <a href="http://www.merl.com/research/license#FoldingNet" target="_blank">FoldingNet</a>.)
```bash
  cd dataset
  bash download_shapenet_core55_catagories.sh
```

### Pre-trained model
You can download the pre-trained models <a href="https://drive.google.com/drive/folders/1XgYWPjAFgn4Vdzm3AjWnGJYFS6Ho9pm5?usp=sharing" target="_blank">here</a>.


### Usage
#### A Minimal Example

We provide an example demonstrating the basic usage in the folder 'mini_example'. 

To visualize the reconstruction from latent capsules with our pre-trained model:
```bash
  cd mini_example/AE
  python viz_reconstruction.py --model ../../checkpoints/shapenet_part_dataset_ae_200.pth
```

To train a point capsule auto encoder with ShapeNetPart dataset by yourself:
```bash
  cd mini_example/AE
  python train_ae.py
```
#### Point Capsule Auto Encoder

To train a point capsule auto encoder with another dataset:
```bash
  cd apps/AE
  python train_ae.py --dataset < shapenet_part, shapenet_core13, shapenet_core55 >
```

To monitor the training process, use TensorBoard by specifying the log directory:
```bash
  tensorboard --logdir log
```
To test the reconstruction accuracy:
```bash
  python test_ae.py  --dataset < >  --model < >
e.g. 
  python test_ae.py --dataset shapenet_core13 --model ../../checkpoints/shapenet_core13_dataset_ae_230.pth
```

To visualize the reconstructed points:
```bash
  python viz_reconstruction.py --dataset < >  --model < >
e.g. 
  python viz_reconstruction.py --dataset shapenet_core13 --model ../../checkpoints/shapenet_core13_dataset_ae_230.pth
```


