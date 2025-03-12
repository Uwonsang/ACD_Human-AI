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
docker pull wilf93/overcooked_plr:v1

# Run the container with GPU support
docker run -it --rm --gpus all --name  <container_name> -v /<host_path>:/<container_path> wilf93/overcooked_plr:v1

```

## Training

```
# activate env
source activate overcooked_plr

# run code (overcooked)
python -m train --env_name Overcooked \
--num_processes=50 \
--level_replay_strategy=return \
--level_replay_score_transform=min \
--level_replay_temperature=0.1 \
--staleness_coef=0.5 \
--log_interval 50 \
--num_env_steps 50000000 \
--use_wandb \
```


## Acknowledgements
This code references by Prioritized Level Replay implementation (https://github.com/facebookresearch/level-replay)
and the Overcooked implementation (https://github.com/HumanCompatibleAI/human_aware_rl)


## License




## Contact
For any questions or feedback, please contact u.wonsang0514@gm.gist.ac.kr.