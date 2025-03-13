# Automatic Curriculum Design for Zero-Shot Human-AI Coordination

The official implementation of the paper [Automatic Curriculum Design for Zero-Shot Human-AI Coordination](https://arxiv.org/abs/2503.07275
).

## Requirements
* Python 3.8
* pytorch 1.8.2
* Other dependencies listed in requirements.txt

## Installation
You can install and run the code either **manually** or using a **Docker container**.

### Option 1: Manual Installation
```
conda create -n overcooked_plr python=3.8
source activate overcooked_plr
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt
bash ./install.sh
```

### Option 2: Using Docker
```
# Pull the Docker image
docker pull wilf93/overcooked_plr:v2

# Run the container with GPU support
docker run -it --rm --gpus all --name  <container_name> -v /<host_path>:/<container_path> wilf93/overcooked_plr:v2
```

## Environment Setup & Preparing
### Activate Environment
```
source activate overcooked_plr
```
### Make human proxy planner
```
python overcooked_pcg/make_planner.py
```


## Training
### Random
```
bash ./run/ppo/plr_random.sh
```

### Prioritized Level Replay (PLR)
```
bash ./run/ppo/plr_td.sh
```

### MAESTRO
```
bash ./run/ppo/plr_random.sh
```

### Our Method
```
bash ./run/ppo/plr_random.sh
```

## Acknowledgements
This code references by Prioritized Level Replay implementation (https://github.com/facebookresearch/level-replay)
and the Overcooked implementation (https://github.com/HumanCompatibleAI/human_aware_rl)


## License
The code in this repository is released under Creative Commons Attribution-NonCommercial 4.0 International License (CC-BY-NC 4.0).


## Contact
For any questions or feedback, please contact u.wonsang0514@gm.gist.ac.kr.