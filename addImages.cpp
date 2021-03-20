
#include <vector>
#include <stdio.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

int main()
{
    cv::Mat image1;
    cv::Mat image2;

    // open images
    image1 = cv::imread("../images/boldt.jpg");
    image2 = cv::imread("../images/rain.jpg");
    if (!image1.data)
        return 0;
    if (!image2.data)
        return 0;

    cv::Mat result;
    // add two images
    cv::addWeighted(image1, 0.7, image2, 0.9, 0., result);
    imwrite("../output/addImages_result_1.jpg", result);

    // using overloaded operator
    result = 0.7 * image1 + 0.9 * image2;
    imwrite("../output/addImages_result_2.jpg", result);

    image2 = cv::imread("../images/rain.jpg", 0);

    // create vector of 3 images
    std::vector<cv::Mat> planes;
    // split 1 3-channel image into 3 1-channel images
    cv::split(image1, planes);
    // add to blue channel
    planes[0] += image2;
    // merge the 3 1-channel images into 1 3-channel image
    cv::merge(planes, result);

    imwrite("../output/addImages_result_3.jpg", result);

    return 0;
}
