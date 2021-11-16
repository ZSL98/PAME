# ETBA
Early Termination in Batched Applications


## Early Exit in Inference

### Prerequisites
Libtorch: Build from source

The following steps are executed in `./Inference`

### Build Prerequisites

    cd ./Inference
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

### Build a sample task (sample_vgg16 for cifar10)
Prepare the cifar10 dataset: Datasets should be stored in the data directory in the root directory of this repository.

    cd ..
    mkdir data
    wget $(url_to_the_dataset_website)
    ...


Build the task project

    cd ./src/sample_vgg16
    mkdir build
    cd build
    cmake ..
    make -j$(nproc)

## imagenet classfication

slzhang @ (fedml) in ~/projects/ETBA/Train/Mytrain on git:main x [22:34:28] C:1
$ python train_imagenet.py

## posenet training

slzhang @ (fedml) in ~/projects/ETBA/Train/human-pose-estimation.pytorch on git:main x [22:59:32] 
$ python pose_estimation/train_exit.py --cfg ./experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml 