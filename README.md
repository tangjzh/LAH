# LAH: Can Robot Only Look Around with Head to Understand Scence?

## QuickStart

Clone the repo and create a new environment:

```shell
git clone https://github.com/tangjzh/LAH
cd LAH

conda env create -f environment.yml
conda activate lah
```

Then, prepare your dataset, please download data from [Matterport3D](https://niessner.github.io/Matterport/) **color images and camera poses** and [labels](https://www.dropbox.com/scl/fi/recc3utsvmkbgc2vjqxur/mp3d_skybox.tar?rlkey=ywlz7zvyu25ovccacmc3iifwe&dl=0), and place your data at `data/` folder.

```shell
├── data
    ├── v1/scan
      ├── 5q7pvUzZiYa
        ├──blip3
        ├──matterport_color_images.zip
        ├──matterport_camera_poses.zip
      ├── 1LXtFkjw3qL
      ├── ....
```

For training, run the following command:

```shell
bash train.sh
```