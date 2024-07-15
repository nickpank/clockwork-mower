#include <iostream>
#include <thread>
#include <string>
#include <vector>
#include <tuple>
#include <type_traits>
#include <stdexcept>
#include <URI.h>
#include <opencv4/opencv2/opencv.hpp>
#include <opencv4/opencv2/features2d.hpp>
#include <opencv4/opencv2/cudastereo.hpp>
#include <opencv4/opencv2/cudawarping.hpp>
#include <jetson-utils/videoSource.h>
#include <jetson-utils/videoOutput.h>
#include <jetson-utils/videoOptions.h>
#include <jetson-utils/cudaOverlay.h>
#include <jetson-utils/cudaMappedMemory.h>
#include "cnpy.h"

using namespace std;
const uint VID_WIDTH = 640; 
const uint VID_HEIGHT = 480; 
void createVideoInputs(videoSource** vidSource0, videoSource** vidSource1, videoOutput** vidOutput, uchar3** arr0, uchar3** arr1, uchar3** arr) {
    videoOptions vidOpts = videoOptions();
    vidOpts.resource = URI("csi://0");;
    vidOpts.deviceType = videoOptions::DeviceType::DEVICE_CSI;
    vidOpts.frameRate = 30;
    vidOpts.width = VID_WIDTH;
    vidOpts.height = VID_HEIGHT;
    vidOpts.flipMethod = videoOptions::FlipMethod::FLIP_ROTATE_180;
    
    *vidSource0 = videoSource::Create(vidOpts);
    vidOpts.resource = URI("csi://1");
    *vidSource1 = videoSource::Create(vidOpts);
    vidOpts.resource = URI("display://0");
    // auto vidOutput0 = videoOutput::Create(vidOpts);
    // vidOpts.resource = URI("display://1");
    // auto vidOutput1 = videoOutput::Create(vidOpts);
    vidOpts.width = VID_WIDTH * 2;
    *vidOutput = videoOutput::Create(vidOpts);

    auto imgSize = sizeof(uchar3) * VID_WIDTH * VID_HEIGHT;
    // *arr0 = (uchar3*)malloc(imgSize);
    // *arr1 = (uchar3*)malloc(imgSize);
    cudaAllocMapped(arr0, imgSize);
    cudaAllocMapped(arr1, imgSize);
    cudaAllocMapped(arr, imgSize*2);
}

struct StereoRemaps {
    cv::cuda::GpuMat wx0;
    cv::cuda::GpuMat wy0;
    cv::cuda::GpuMat wx1;
    cv::cuda::GpuMat wy1;
};

void loadRemaps(StereoRemaps*& remaps) {
    // Data is relative to build/ folder
    cv::FileStorage fs("stereo_remaps_zoomed.yml", cv::FileStorage::READ);
    cv::Mat wx0Temp, wy0Temp, wx1Temp, wy1Temp;
    fs["wx0"] >> wx0Temp;
    fs["wy0"] >> wy0Temp;
    fs["wx1"] >> wx1Temp;
    fs["wy1"] >> wy1Temp;
    remaps->wx0.upload(wx0Temp);
    remaps->wy0.upload(wy0Temp);
    remaps->wx1.upload(wx1Temp);
    remaps->wy1.upload(wy1Temp);
}

// void rectify()

int main() {
    videoSource* vidSource0;
    videoSource* vidSource1;
    videoOutput* vidOutput;
    uchar3* arr0, *arr0_aux;
    uchar3* arr1, *arr1_aux;
    uchar3* arr;
    size_t imgSize = sizeof(uchar3) * VID_WIDTH * VID_HEIGHT;
    cudaAllocMapped(&arr0_aux, imgSize);
    cudaAllocMapped(&arr1_aux, imgSize);
    
    createVideoInputs(&vidSource0, &vidSource1, &vidOutput, &arr0, &arr1, &arr);
    
    StereoRemaps* remaps = new StereoRemaps();
    loadRemaps(remaps);

    int status0;
    int status1;
    while (true) {
        // // Simple capture & sync: Overlay them onto the same array and then cudaCrop when separation is needed
        // vidSource0->Capture(&arr0, &status0);
        // vidSource1->Capture(&arr1, &status1);
        // CUDA(cudaOverlay(arr0, VID_WIDTH, VID_HEIGHT, arr, VID_WIDTH*2, VID_HEIGHT, 0, 0));
        // CUDA(cudaOverlay(arr1, VID_WIDTH, VID_HEIGHT, arr, VID_WIDTH*2, VID_HEIGHT, VID_WIDTH, 0));
        // vidOutput->Render(arr, VID_WIDTH*2, VID_HEIGHT);
        // // vidOutput0->Render(arr0, 640, 480);
        // // vidOutput1->Render(arr1, 640, 480);



        // Threaded capture & sync: Same overlay + cudaCrop
        std::thread imgThread0([vidSource0, &arr0, &status0]() {vidSource0->Capture(&arr0, &status0);});
        std::thread imgThread1([vidSource1, &arr1, &status1]() {vidSource1->Capture(&arr1, &status1);});
        imgThread0.join();
        imgThread1.join();
        imgThread0.~thread();
        imgThread1.~thread();

        // Keep the images on the GPU and flip flop b/w them
        cv::cuda::GpuMat gpuMat0(VID_HEIGHT, VID_WIDTH, CV_8UC3, arr0);
        cv::cuda::GpuMat gpuMat1(VID_HEIGHT, VID_WIDTH, CV_8UC3, arr1);
        cv::cuda::GpuMat gpuMat0_aux(VID_HEIGHT, VID_WIDTH, CV_8UC3, arr0_aux);
        cv::cuda::GpuMat gpuMat1_aux(VID_HEIGHT, VID_WIDTH, CV_8UC3, arr1_aux);

        cv::cuda::remap(gpuMat0, gpuMat0_aux, remaps->wx0, remaps->wy0, cv::INTER_LINEAR);
        cv::cuda::remap(gpuMat1, gpuMat1_aux, remaps->wx1, remaps->wy1, cv::INTER_LINEAR);

        CUDA(cudaOverlay((uchar3*)gpuMat0_aux.data, VID_WIDTH, VID_HEIGHT, arr, VID_WIDTH*2, VID_HEIGHT, 0, 0));
        CUDA(cudaOverlay((uchar3*)gpuMat1_aux.data, VID_WIDTH, VID_HEIGHT, arr, VID_WIDTH*2, VID_HEIGHT, VID_WIDTH, 0));
        vidOutput->Render(arr, VID_WIDTH*2, VID_HEIGHT);
    }
    vidSource0->Close();
    vidSource1->Close();
    return 0;
}