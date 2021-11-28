#ifndef CHECK_H
#define CHECK_H

#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

std::vector<int> check_on_cpu(std::shared_ptr<nvinfer1::ICudaEngine> engine,
                              std::string model_name, 
                              float* exitPtr_host,
                              size_t batch_size_s1_){
    if (model_name == "resnet"){
        auto dims = engine->getBindingDimensions(2);
        float exit_output[dims.d[0]][dims.d[1]];
        // for (int i = 0; i < dims.d[0]; i++){
        //     for (int j = 0; j < dims.d[1]; j++){
        //         exit_output[i][j] = exitPtr_host[i*dims.d[1]+j];
        //     }
        // }
        for (int i = 0; i < dims.d[0]*dims.d[1]; i++){
            exit_output[i/dims.d[1]][i%dims.d[1]] = exitPtr_host[i];
        }
        // std::cout << dims.d[0] << "====++++====" << dims.d[1] << std::endl;
        std::vector<int> copy_list;
        for (int i = 0; i < dims.d[0]; i++){
            copy_list.push_back(1);
        }
        return copy_list;
    }
}

#endif