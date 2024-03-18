# Reinforcement Learning for Quadruped #
This package is forked from [Extreme Parkour](https://github.com/chengxuxin/extreme-parkour)
The packege here was modified to use for a Unitree Go1 to perform weaving around poles using Reinforcement learning. 

[!running_video](https://github.com/sdalal1/extreme-parkour-barkour/assets/80363654/84d8e007-ffa6-4e76-8c18-920cc7423f1f)


### Installation ###
```bash
# Download the Isaac Gym binaries from https://developer.nvidia.com/isaac-gym 
# Originally trained with Preview3, but haven't seen bugs using Preview4.
# Prefer running the package inside the Docker Coontainer Provided in this package
# Copy the docker file to the isaacgym/docker
cp Dockerfile_for_Isaacgym isaacgym/docker/Dockerfile
# Run the ./build.sh and ./run.sh
cd isaacgym/docker
./build.sh && ./run.sh
docker exec -it isaacgym_container /bin/bash
#The docker should open as a root and follow the next steps for setup
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
git clone git@github.com:sdalal1/extreme-parkour-barkour.git
cd extreme-parkour-barkour
cd isaacgym/python && pip install -e .
cd ~/extreme-parkour/rsl_rl && pip install -e .
cd ~/extreme-parkour/legged_gym && pip install -e .
pip install "numpy<1.24" pydelatin wandb tqdm opencv-python ipdb pyfqmr flask
```

### Usage ###
The scripts folder contains all the required scripts to run the training and playback of the policy. Follow the steps below.

`cd legged_gym/scripts`
1. Train base policy:  
```bash
python train.py --exptid xxx-xx-WHATEVER --device cuda:0
```
Train 10-15k iterations (8-10 hours on 3090) (at least 15k recommended).

2. Train distillation policy:
```bash
python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera
```
Train 5-10k iterations (5-10 hours on 3090) (at least 5k recommended). 
>You can run either base or distillation policy at arbitary gpu # as long as you set `--device cuda:#`, no need to set `CUDA_VISIBLE_DEVICES`.

3. Play base policy:
```bash
python play.py --exptid xxx-xx
```
No need to write the full exptid. The parser will auto match runs with first 6 strings (xxx-xx). So better make sure you don't reuse xxx-xx. Delay is added after 8k iters. If you want to play after 8k, add `--delay`

4. Play distillation policy:
```bash
python play.py --exptid yyy-yy --delay --use_camera
```
[training_video](https://github.com/sdalal1/extreme-parkour-barkour/assets/80363654/8a04eaa2-7398-49d1-8855-cff785768130)


5. Save models for deployment:
```bash
python save_jit.py --exptid xxx-xx
```
This will save the models in `legged_gym/logs/parkour_new/yyy-yy/traced/`.

### Saved Model
There is a saved model for deployment in the logs folder. The model is `go-dist-2.0.7`. Thos saved model was used for weaving poles.
The depth policy of the model was deployed using the ROS package.

### ROS package for deployment
There is a ROS package included in the bundle. The saved model is in there. 
```bash
#connect to the unitree with following commands
ifconfig # Tells you enpxxx, your computer's network interfaces 
sudo ifconfig enpxxx down
sudo ifconfig enpxxx 192.168.123.162/24
sudo ifconfig enpxxx up
ping 192.168.123.161
```
The saved model was ran on an external jetson-nano which was connected to a Realsense Camera.
```bash
#Launch the realsense ros node
ros2 launch realsense_camera rs_launch.py
```

```bash
colcon build
source install/setup.bash
ros2 launch hallway_detection detect.launch.xml
```

### Viewer Usage
Can be used in both IsaacGym and web viewer.
- `ALT + Mouse Left + Drag Mouse`: move view.
- `[ ]`: switch to next/prev robot.
- `Space`: pause/unpause.
- `F`: switch between free camera and following camera.

### Arguments
- --exptid: string, can be `xxx-xx-WHATEVER`, `xxx-xx` is typically numbers only. `WHATEVER` is the description of the run. 
- --device: can be `cuda:0`, `cpu`, etc.
- --delay: whether add delay or not.
- --checkpoint: the specific checkpoint you want to load. If not specified load the latest one.
- --resume: resume from another checkpoint, used together with `--resumeid`.
- --seed: random seed.
- --no_wandb: no wandb logging.
- --use_camera: use camera or scandots.


### Citation
```
@article{cheng2023parkour,
title={Extreme Parkour with Legged Robots},
author={Cheng, Xuxin and Shi, Kexin and Agarwal, Ananye and Pathak, Deepak},
journal={arXiv preprint arXiv:2309.14341},
year={2023}
}
```