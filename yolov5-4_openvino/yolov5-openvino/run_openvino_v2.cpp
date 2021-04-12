#include <iostream>
#include <vector>

#include <inference_engine.hpp>
#include <ngraph/ngraph.hpp>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace InferenceEngine;
using namespace cv;


int main()
{
    Core ie;
    std::vector<std::string> availableDevices = ie.GetAvailableDevices();
    for (int i = 0; i < availableDevices.size(); i++) {
        printf("supported device name : %s \n", availableDevices[i].c_str());
    }

    auto cnnNetwork = ie.ReadNetwork("../weights/yolov5s.onnx");
    cv::Mat inframe = cv::imread("../data/images/bus.jpg")

    //
    // 设置输入格式
    input_info = cnnNetwork.input_info
    for (auto &item : input_info) {
        auto input_data = item.second;
        input_data->setPrecision(Precision::FP32);
        input_data->setLayout(Layout::NCHW);
        input_data->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
        input_data->getPreProcess().setColorFormat(ColorFormat::RGB);
    }

    // 设置输出格式
    for (auto &item : output_info) {
        auto output_data = item.second;
        output_data->setPrecision(Precision::FP32);
    }
    auto executable_network = ie.LoadNetwork(network, "CPU");

    //


    //从模型中获取输入数据的格式信息
    InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
    InputInfo::Ptr& input = inputInfo.begin()->second;
    _input_name = inputInfo.begin()->first;
    input->setPrecision(Precision::FP32);
    input->getInputData()->setLayout(Layout::NCHW);
    ICNNNetwork::InputShapes inputShapes = cnnNetwork.getInputShapes();
    SizeVector& inSizeVector = inputShapes.begin()->second;
    cnnNetwork.reshape(inputShapes);
    //从模型中获取推断结果的格式
    _outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
    for (auto &output : _outputinfo) {
        output.second->setPrecision(Precision::FP32);
    }
    //获取可执行网络,这里的CPU指的是推断运行的器件,可选的还有"GPU",这里的GPU指的是intel芯片内部的核显
    //配置好核显所需的GPU运行环境,使用GPU模式进行的推理速度上有很大提升,这里先拿CPU部署后面会提到GPU环境的配置方式
    _network =  ie.LoadNetwork(cnnNetwork, "CPU");

    InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();
    Blob::Ptr frameBlob = infer_request->GetBlob(_input_name);
    InferenceEngine::LockedMemory<void> blobMapped = InferenceEngine::as<InferenceEngine::MemoryBlob>(frameBlob)->wmap();
    float* blob_data = blobMapped.as<float*>();
    //nchw
    for(size_t row =0;row<640;row++){
        for(size_t col=0;col<640;col++){
            for(size_t ch =0;ch<3;ch++){
                //将图像转换为浮点型填入模型
                blob_data[img_size*ch + row*640 + col] = float(inframe.at<Vec3b>(row,col)[ch])/255.0f;
            }
        }
    }

    infer_request->Infer();

    return 0;
}
