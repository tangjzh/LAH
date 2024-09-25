# LAH: Can Robot Only Look Around with Head to Understand Scence?

## QuickStart

### Install from Docker

- Prerequisites:
  - Docker installed on your machine.
  - NVIDIA Docker for CUDA support.

Clone the repo and create a docker container:

```shell
git clone https://github.com/tangjzh/LAH
cd LAH

docker build -t lah .
docker run --gpus all -it lah
```

### Install from Source

Clone the repo and create a new environment:

```shell
git clone https://github.com/tangjzh/LAH
cd LAH

conda env create -f environment.yml
conda activate lah
```

### Training

Then, prepare your dataset, please download data from [Matterport3D](https://niessner.github.io/Matterport/) **color images and camera poses** and [labels](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0), and place your data at `data/` folder.

You also need to download our released data of EmbodiedScan and VLN-CE, and place them at `data/` folder.

```shell
├── data
    ├── v1/scan
      ├── 5q7pvUzZiYa
        ├──blip3
        ├──matterport_color_images.zip
        ├──matterport_camera_poses.zip
      ├── 1LXtFkjw3qL
      ├── ....
    ├── embodiedscan
    ├── vlnce
```

For training, run the following command:

```shell
# train.sh, you can edit this for your need.
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# export WORLD_SIZE=4
# export MASTER_ADDR='localhost'
# export MASTER_PORT=25002
# export LOCAL_RANK=4

# torchrun --nproc_per_node=$WORLD_SIZE train.py

bash train.sh
```