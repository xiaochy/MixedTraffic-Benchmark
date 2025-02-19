# MixedTraffic Benchmark
Official code for the paper:
> **Optimizing Efficiency of Mixed Traffic through Reinforcement Learning: A Topology-Independent Approach and Benchmark**
>
> Chuyang Xiao, Dawei Wang, Xinzheng Tang, Jia Pan, Yuexin Ma
>
> <a href='https://arxiv.org/abs/2501.16728'><img src='https://img.shields.io/badge/arXiv-2301.05294-red'></a> <a href='https://sites.google.com/berkeley.edu/mixedtrafficplus?usp=sharing'><img src='https://img.shields.io/badge/Project-Video-Green'></a>

## Environment Set-up
### Requirements -- Anaconda Set-up
We highly recommend installing on Ubuntu 22.04 as this version of Ubuntu has been tested. If you have a Windows machine, we recommend using WSL to create an Ubuntu 22.04 virtual machine for setting up the repo.
```
ubuntu
```
```
conda create -n MixedTrafficPlus python=3.8
conda activate MixedTrafficPlus
conda env update -f MixedTrafficPlus.yml
```
### Install SUMO
```
sudo apt-get install sumo sumo-tools sumo-doc
```

## Training
- To train on various topologies which involve intersections and roundabouts, you can run the following command and choose the rv_rate you want to use as the folder name.
```
--resume_dir default="checkpoint". -> It will automatically find the latest checkpoint in this checkpoint folder
--wandb-id default="". -> It will automatically create a new wandb run witha new wandb_id
```
```
cd benchmark_training
cd large_rv={rv_rate}
python train_large_benchmark.py --resume-dir="the/path/to/the/checkpoint/you/want/to/resume" --wandb-id "wandb-id_you_want_to_use"

```

- To train on only one topologies, for example, roundabouts, you can run the following command with rv_rate=1.
```
cd our_roundabout
python train_our_ra.py --resume-dir="the/path/to/the/checkpoint/you/want/to/resume" --wandb-id "wandb-id_you_want_to_use"
```

## Testing
- To test on various topologies which involve intersections and roundabouts, you can run the following command and choose the rv_rate you want to use as the folder name.
```
cd benchmark_training
cd large_rv={rv_rate}
python test_large_benchmark.py --resume-dir="the/path/to/the/checkpoint/you/want/to/resume" --wandb-id "wandb-id_you_want_to_use"
```

- To test on only one topologies, for example, roundabouts, you can run the following command with rv_rate=1.
```
cd our_roundabout
python test_our_ra.py --resume-dir="the/path/to/the/checkpoint/you/want/to/resume" --wandb-id "wandb-id_you_want_to_use"
```

## Generate More Data
You can put your osm files downloaded from openstreetmap.org to train_file or test_file folder and select the folder difficulty(mode = 400 veh/h, easy(1000 veh/h), middle(3000 veh/h), hard(5000 veh/h)). Then you can run
``` 
./scripts/generate_train_set/generate_{mode}_routes.sh 
```
if you want to generate training scenario files.





