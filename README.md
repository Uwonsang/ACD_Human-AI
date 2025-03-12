## Installation
```
conda create -n overcooked_plr python=3.8
source activate overcooked_plr
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -r requirements.txt
bash ./install.sh
```
If not install the procgen env, `apt-get update`, `apt-get install build-essential`

## Examples
### Train PPO with value-based level reply with rank prioritization on BigFish

```
# run docker
docker run -it --rm --name  <container_name> -v /<host_path>:/<container_path> <image_name>

# activate env
source activate overcooked_plr

# run code (bigfish)
python -m train --env_name bigfish \
--num_processes=64 \
--level_replay_strategy='value_l1' \
--level_replay_score_transform='rank' \
--level_replay_temperature=0.1 \
--staleness_coef=0.1

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

