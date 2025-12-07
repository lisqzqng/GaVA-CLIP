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

Download the **[pretrained CLIP checkpoint](https://seafile.unistra.fr/f/42a0de36d9fc43979c82/)**  and place under the `pretrained directory`.

### Training Instruction
# 1. Data preparation
Please follow the instructions in **[TULIP dataset repository](https://zenodo.org/records/14199925)** to donwload gait videos of 11 Subjects. 
And follow the data preprocessinig:
```
(To be added)
```

Prepare the initial continuous per-class textual embeddings:
```bash
wget --content-disposition "https://drive.google.com/uc?export=download&id=1ncddV8M8p_yzUjztmECKK-LgJQH-WWiw"
tar -xf data.tar.xz
rm data.tar.xz
```

# 2. Launch the training and validation
Here, we provide the code for leave-one-subject-out cross-validation on the  [TULIP data v1](https://zenodo.org/records/14199925).
```bash
source train_scripts/updrs_3cls_train_tulip.sh
``` 
The output models and results can be found in in `./logs/`.

Specifically, we exclude Subject 1 from the available 11 subjects. The cross-validation is conducted 10-fold. In each fold: we use 9 subjects for training and thr remaining 1 subject for validation (a different subject held out each fold).
The training scripts are configured for an Nvidia RTX 6000 Ada GPU.



# 3. Cross-validation results compared with SOTA.
Comparing the Gait Scoring performance of GaVA-CLIP with SOTA methods on 10 subjects of [TULIP data v1](https://zenodo.org/records/14199925). Evaluation metrics are top-1 accuracy (%), F1-
score, precision rate (%), recall rate (%), and (classwise) weighted F1-score.
|Method                |Accuracy| F1-score| Precision| Recall| weighted F1-score|
|-----------------|----------:|-------:|-------:|-------:|-------------:|
OF-DDNet            |83.22   |   0.828   |    84.26   |    85.32   |    0.821 |
KShapeNet           |67.86   |    0.638   |    73.44   |    64.45   |    0.650 |
GaitForeMer         | 74.53   |    0.747   |    75.13   |    74.53   |    0.743 |
Vita-CLIP              |87.88   |    0.879   |    88.27   |    88.24   |    0.886 |
GaVA-CLIP (Ours)| 92.29   |    0.917   |    92.50   |    91.28  | 0.921 |

# References
For the SOTA methods, we greatly benefit from the following resources:
+ [OF-DDNet](https://github.com/mlu355/PD-Motor-Severity-Estimation).
+ [KShapNet](https://github.com/MLMS-CG/AD-DLB-Classifier).
+ [GaitForeMer](https://github.com/markendo/GaitForeMer).
+ [Vita-CLIP](https://github.com/TalalWasim/Vita-CLIP).

## Acknowledgements
Our code is based on [Vita-CLIP](https://github.com/TalalWasim/Vita-CLIP), and the continuous per-class textual embeddings are encoded by [KEPLER](https://github.com/THU-KEG/KEPLER).
