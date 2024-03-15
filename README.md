# PAME: Precision-Aware Multi-Exit DNN Serving for Reducing Latencies of Batched Inferences
This is the code repository of the ICS'22 paper: 
PAME: Precision-Aware Multi-Exit DNN Serving for Reducing Latencies of Batched Inferences
The key concept of this work is to add and train exits for DNN networks. Then decide for samples to choose exit paths at the inference stage.

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

## 1 Train Exits with Transfer Learning

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

---

## 2 Determine the to-be-used Exits

Now let us begin to determine the exits one by one. We start from determining the first exit.

### 2.1 Determine the First Exit

#### Step 1: Characterizing the Load

We first determine the thresholds of exits according to the load (Precision-aware Candidate Configuration). Then we obtain the Batching Pattern Matrix with the thresholds of exits fixed.

##### Precision-aware Candidate Configuration
In this step, we determine the thresholds in each exit candidate. The optimal thresholds are determined using grid searching. The grid searching process is integrated in `./Train/metric_convert.py` and can be launched by setting `--init True`, otherwise the grid searching process is skipped. The intermediate results of grid searching are stored in `./Train/conversion_results`

The optimal thresholds is then determined according to the intermediate results and is recorded in `./Train/opt_thres_record`

For example, with the backbone of a complete resnet and 99% precision tolerance (allowing 1% precision degradation), run the command below to characterize the load of Imagenette dataset with batch size 32:

    cd Train
    python metric_convert.py --task resnet --dataset_name imagenette --metric_thres 99 --batch_size 32 --last_exit 0 --init True

If the grid searching is finished under this set of configurations and you only wish to modify the metric threshold, you can omit the `--init` arg and rerun. Any other changes to the configurations require the `--init` to be True.

##### Obtain the Batching Pattern Matrix

After the determination of exit candidates' configuration, the traces of samples are recorded in `./Train/moveon_dict`. The Batching Pattern Matrix is then obtained by running `cd Train && python pick_exit.py`

#### Step 2: Characterizing the Inference

We characterize the inference with the Inference Time Matrix.

##### Obtain the Inference Time Matrix

    cd ./Inference/src/exit_placement
    mkdir results
    mkdir build && cd build
    cmake ..
    make -j

Edit the configuration file `profiler_config.json`, and run `./out/sample <task_name>` to obtain the Inference Time Matrix which is stored at `./results`. In this tutorial's example, to get the Inference Time Matrix of ResNet in image classification, just run `./out/sample resnet`

### 2.2 Find More Exits

Once you have determined an exit, let's say, an exit inserted after the 7th block in this example, you can continue to find other posterior exits by characterizing the load:

    python metric_convert.py --task resnet --dataset_name imagenette --metric_thres 99 --batch_size 32 --last_exit 0 7 --init True

as well as characterizing the inference.

## 3 Implementation

    cd ./Inference/src/run_engine
    mkdir build && cd build
    cmake ..
    make -j

Run `./out/sample <task_name>` to see results.
