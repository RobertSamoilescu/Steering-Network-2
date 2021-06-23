# Steering-Network-2

<p align='center'>
  <img src='sample/sample1.png' alt='sample 1' width=1024/>
</p>

<p align='center'>
  <img src='sample/sample2.png' alt='sample 2' width=1024/>
</p>

## Pre-requisits
```shell
mkdir -p pipeline/models/monodepth
cd pipeline/models/monodepth
```
For monodepth, download the pre-trained models from <a href='https://drive.google.com/drive/folders/18kTR4PaRlQIeEFJ2gNkiXYnFcTfyrRNH?usp=sharing'>here</a>

```shell
mkdir -p pipeline/models/inpaint
cd pipeline/models/inpaint
```
For the inpaint, download the pre-trained model from <a href='https://drive.google.com/drive/folders/1oeVxVnR5BIZ1QM-ClY6Xa4CogxTQzmZx?usp=sharing'>here</a>

```shell
mkdir -p pipeline/models/flow
cd pipeline/models/flow
```
For optical flow, download the pre-trained model from <a href='https://drive.google.com/drive/folders/1sahN3m6salz64fG8XFGuA0vYklkWYMNu?usp=sharing'>here</a>


## Create dataset

```shell
mkdir raw_dataset
```

* Download the UBP dataset into the "raw_dataset" directory. A sample of the UPB dataset is available <a href="https://drive.google.com/drive/folders/1p_2-_Xo-Wd9MCnkYqPfGyKs2BnbeApqn?usp=sharing">here</a>.

```shell
mkdir scene_splits
```

* Download the scene splits into the "scene_splits" directory. The train-validation split is available <a href="https://github.com/RobertSamoilescu/UPB-Dataset-Split">here</a>.
In the "scene_splits" directory you should have: "train_scenes.txt" and "test_scenes.txt".

```
# create the dataset
python -m scripts.create_dataset --src_dir raw_dataset --dst_dir dataset

# split the dataset into train-test
python -m scripts.split_dataset --split_dir scene_splits

# compute balancing weights
python -m scripts.weights
```

## Train models
```shell
./run_train.sh
```

## Test models - Open-loop evaluation
```shell
./run_test.sh
```

<!--
## Results - Open-loop evaluation
| Model  | Speed | Stacked    | Disp(aux)  | Flow(aux) | MEAN | STD  | MIN  | MAX   |
| ------ | ----- | ---        | ---------- | --------- | ---- | ---- | ---- | ----  |
| SIMPLE | YES   | NO         | NO         | NO        | 0.755| 1.008| 0.003|14.118 |
| RESNET | YES   | NO         | NO         | NO        | 0.711| 1.304| 0.001|11.217 |
| SIMPLE | YES   | NO         | YES        | NO        | 0.755| 0.996| 0.002|15.949 |
| RESNET | YES   | NO         | YES        | NO        | 0.684| 1.266| 0.001|13.099 |
| SIMPLE | YES   | NO         | YES        | YES       | 0.443| 0.859| 0.001|16.358 |
| RESNET | YES   | NO         | YES        | YES       | 0.426| 0.870| 0.000|10.500 |
| Model  | Speed | Stacked    | Disp(aux)  | Flow(aux) | MEAN | STD  | MIN  | MAX  |
| ------ | ----- | ---        | ---------- | --------- | ---- | ---- | ---- | ---- |
| SIMPLE | NO    | YES        | NO         | NO        | 0.470| 0.884| 0.001|15.600|
| RESNET | NO    | YES        | NO         | NO        | 0.430| 0.945| 0.002|12.256|
| SIMPLE | NO    | YES        | YES        | NO        | 0.503| 1.036| 0.000|18.207|
| RESNET | NO    | YES        | YES        | NO        | 0.418| 0.921| 0.000|12.379|
| SIMPLE | NO    | YES        | YES        | YES       | 0.457| 0.820| 0.001|15.430|
| RESNET | NO    | YES        | YES        | YES       | 0.435| 0.953| 0.000|11.929|
-->
