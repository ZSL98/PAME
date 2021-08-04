import onnx
from onnx import helper
import sys,getopt
import torch
import torchvision
from transformers import AutoTokenizer, AutoModel

from pathlib import Path
from transformers.convert_graph_to_onnx import convert

#加载模型
def loadOnnxModel(path):
    model = onnx.load(path)
    return model

#获取节点和节点的输入输出名列表，一般节点的输入将来自于上一层的输出放在列表前面，参数放在列表后面
def getNodeAndIOname(nodename,model):
    for i in range(len(model.graph.node)):
        if model.graph.node[i].name == nodename:
            Node = model.graph.node[i]
            input_name = model.graph.node[i].input
            output_name = model.graph.node[i].output
    return Node,input_name,output_name

#获取对应输入信息
def getInputTensorValueInfo(input_name,model):
    in_tvi = []
    for name in input_name:
        for params_input in model.graph.input:
            if params_input.name == name:
               in_tvi.append(params_input)
        for inner_output in model.graph.value_info:
            if inner_output.name == name:
                in_tvi.append(inner_output)
    return in_tvi

#获取对应输出信息
def getOutputTensorValueInfo(output_name,model):
    out_tvi = []
    for name in output_name:
        out_tvi = [inner_output for inner_output in model.graph.value_info if inner_output.name == name]
        if name == model.graph.output[0].name:
            out_tvi.append(model.graph.output[0])
    return out_tvi

#获取对应超参数值
def getInitTensorValue(input_name,model):
    init_t = []
    for name in input_name:
        init_t = [init for init in model.graph.initializer if init.name == name]
    return init_t

#构建单个节点onnx模型
def createSingelOnnxModel(ModelPath,nodename,SaveType="",SavePath=""):
    model = loadOnnxModel(str(ModelPath))
    Node,input_name,output_name = getNodeAndIOname(nodename,model)
    in_tvi = getInputTensorValueInfo(input_name,model)
    out_tvi = getOutputTensorValueInfo(output_name,model)
    init_t = getInitTensorValue(input_name,model)

    graph_def = helper.make_graph(
                [Node],
                nodename,
                inputs=in_tvi,  # 输入
                outputs=out_tvi,  # 输出
                initializer=init_t,  # initalizer
            )
    model_def = helper.make_model(graph_def, producer_name='onnx-example')
    print(nodename+"onnx success!")

#获取节点数量
def getNodeNum(model):
    return len(model.graph.node)

#获取节点类型
def getNodetype(model):
    op_name = []
    for i in range(len(model.graph.node)):
        if model.graph.node[i].op_type not in op_name:
            op_name.append(model.graph.node[i].op_type)
    return op_name

#获取节点名列表
def getNodeNameList(model):
    NodeNameList = []
    for i in range(len(model.graph.node)):
        NodeNameList.append(model.graph.node[i].name)
    return NodeNameList

#获取模型的输入信息
def getModelInputInfo(model):
    return model.graph.input[0]

#获取模型的输出信息
def getModelOutputInfo(model):
    return model.graph.output[0]

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # model = AutoModel.from_pretrained("bert-base-uncased")
    # input_names = [ "input_1" ]
    # output_names = [ "output_1" ]

    # inputs = tokenizer("Hello world!", return_tensors="pt")
    # torch.onnx.export(model, inputs, "./models/bert-base.onnx", verbose=True, input_names=input_names, output_names=output_names)

    convert(framework="pt", model="bert-base-cased", output=Path("models/bert-base/bert-base-cased.onnx"), opset=11)

    # model = loadOnnxModel("./models/vgg16.onnx")
    # print(getNodeNum(model))