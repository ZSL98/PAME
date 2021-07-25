# ETBA
Early Termination in Batched Applications


## Early Exit in Inference
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