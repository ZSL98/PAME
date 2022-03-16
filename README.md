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

## Train exits with transfer learning

### ImageNet classfication

First prepare the dataset in the dataset directory `<path to imagenet>`, which contains the `train` directory and the `val` directory. To accelerate the training process, we utilize [Nvidia's deeplearning examples](https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets) and modify the codes within.

    cd ./Train-Nvidia
    mkdir checkpoints
    python ./main.py <path to imagenet>

The model checkpoints would be stored in the `checkpoint` directory. One can also train without Nvidia GPUs by running the following commands.

    cd ./Train/Mytrain
    mkdir checkpoints
    python train_imagenet.py

### Pose Estimation

    cd ./Train/pose_estimation
    mkdir checkpoints
    python pose_estimation/train_exit.py --cfg ./experiments/mpii/resnet101/384x384_d256x3_adam_lr1e-3.yaml
    
### Semantic Segmentation

    cd ./Train/openseg
    bash ./train_with_exit.sh

### Language Models

    cd ./Train/bert
    bash ./scripts/train_glue.sh

## Precision-aware Candidate Configuration

## Obtain the Inference Time Matrix

    cd ./Inference/src/exit_placement
    mkdir build && cd build
    cmake ..
    make -j

Edit the configuration file `profiler_config.json`, and run `./out/sample <task_name>` to obtain the Inference Time Matrix which is stored at `./results`. For example, to get the Inference Time Matrix of ResNet in image classification, just run `./out/sample resnet`



