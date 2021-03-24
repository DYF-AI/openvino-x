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

    return 0;
}