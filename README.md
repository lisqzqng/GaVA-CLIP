# GaVA-CLIP: Refining Multimodal Representations with Clinical Knowledge and Numerical Parameters for Gait Video Analysis in Neurodegenerative Diseases


## Environment Setup

Use the docker image and Dockfile to set up the environment, make ensure that you modify the user name and all the directories in `Dockerfile`, and modify the volume path in `docker_script.sh`:
```bash
docker build -t gavaclip .
source docker_script.sh
```
To access the container (the following model training is to be done inside the docker container):
```bash
docker exec -it test1 bash
```

## Supervised Training

### Dataset Preparation
We expect that `--train_list_path` and `--val_list_path` command line arguments to be a data list file of the following format
```
<path_1>,<label_1>
<path_2>,<label_2>
...
<path_n>,<label_n>
```
where `<path_i>` points to a video file, and `<label_i>` is an integer between `0` and `num_classes - 1`.
`--num_classes` should also be specified in the command line argument.

Additionally, `<path_i>` might be a relative path when `--data_root` is specified, and the actual path will be
relative to the path passed as `--data_root`.

The class mappings in the open-source weights are provided at [Kinetics-400 class mappings](data/k400_class_mappings.json)

### Download Pretrained CLIP Checkpoint

Download the **[pretrained CLIP checkpoint](https://drive.google.com/file/d/17xSat9ZqL8p3RjpfTdqjxrBfcwZgZ2OE/view?usp=sharing)**  and place under the `pretrained directory`.

### Training Instruction
# 1. Data preparation
Donwload the example TULIP 10-subjects data:
```bash
mkdir -p datasets
cd datasets
wget --content-disposition "https://seafile.unistra.fr/f/3ef03cc3a9394d7a9c48/?dl=1"
tar -xf tulip.tar.xz
rm tulip.tar.xz
cd ../
```
Prepare the initial continuous per-class textual embeddings:
```bash
wget --content-disposition "https://seafile.unistra.fr/f/d90aa57bc07b434f9f56/?dl=1"
tar -xf data.tar.xz
rm data.tar.xz
```

# 2. Launch the training and validation
To train GaVA-CLIP in a 10-fold cross-validation manner using TULIP datset on Nvidia GeForce RTX 3090:
```bash
source train_scripts/updrs_3cls_train_tulip.sh
``` 
The output models and results can be found in in `./logs/`.



## Acknowledgements
Our code is based on [Vita-CLIP](https://github.com/TalalWasim/Vita-CLIP), and the continuous per-class textual embeddings are encoded by [KEPLER](https://github.com/THU-KEG/KEPLER).