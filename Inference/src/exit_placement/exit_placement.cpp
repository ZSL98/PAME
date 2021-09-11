/*
Exit placement
*/

#include <cuda_runtime.h>
#include "exit_placement.h"

bool Profiler::build()
{

}

bool Profiler::infer()
{

}

ProfilerConfig getInstConfig(const rapidjson::Document& config_doc)
{
    assert(config_doc["seg_files"].IsArray());
    const rapidjson::Value& model_files = config_doc["seg_files"];
    const rapidjson::Value& thresholds = config_doc["thresholds"];
    assert(model_files.Size() == config_doc["seg_num"].GetUint());
    ProfilerConfig profiler_config{config_doc["seg_num"].GetUint(),
        config_doc["ee_num"].GetUint(), config_doc["dir"].GetString(),
        nvinfer1::DataType::kFLOAT, config_doc["fp16"].GetBool()};
    for (rapidjson::SizeType i = 0; i < model_files.Size(); i++) {
        profiler_config.setSegFileName(i, model_files[i].GetString());
    }
    for (rapidjson::SizeType i = 0; i < thresholds.Size(); i++) {
        profiler_config.ee_thresholds_[i] = float(thresholds[i].GetDouble());
    }
    return profiler_config;
}

int main(int argc, char** argv)
{
    std::string config_path = "/home/slzhang/projects/ETBA/Inference/src/exit_placement/profiler_config.json";
    FILE* config_fp = fopen(config_path.c_str(), "r");
    if(!config_fp){
        std::cout<<"failed to open config.json"<<endl;
        return -1;
    }
    char read_buffer[65536];

    rapidjson::FileReadStream config_fs(
        config_fp, read_buffer, sizeof(read_buffer));
    rapidjson::Document config_doc;
    config_doc.ParseStream(config_fs);
    ProfilerConfig profiler_config = getInstConfig(config_doc);
    Profiler inst = Profiler(
        profiler_config, config_doc["min_bs"].GetUint(),
        config_doc["opt_bs"].GetUint(), config_doc["max_bs"].GetUint(), config_doc["bs_num"].GetUint(),
        nvinfer1::ILogger::Severity::kINFO);
}