
#include "utils.hpp"

#include <cstdlib>
#include <random>
#include <algorithm>

using namespace std;
using namespace cv;

float* read_image(string path, int* h, int* w) {
    cv::Mat img = cv::imread(path, IMREAD_GRAYSCALE);
    *h = img.rows;
    *w = img.cols;

    float* data = (float*) malloc(img.cols*img.rows*sizeof(float));
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            data[j+img.cols*i] = img.at<uint8_t>(i, j) / 255.0;
        }
    }
    return data;
}

void show_image(float* data, int h, int w) {
    Mat img(h, w, CV_32FC1, data);
    imshow("image", img);
    waitKey(0);
}

void array_add(float* arr1, float* arr2, float* res, int size) {
    for (int i = 0; i < size; i++) {
        res[i] = arr1[i] + arr2[i];
        // clip to [0,1]
        res[i] = std::min(res[i], 1.0f);
        res[i] = std::max(res[i], 0.0f);
    }
}

void array_subtract(float* arr1, float* arr2, float* res, int size) {
    for (int i = 0; i < size; i++) {
        res[i] = arr1[i] - arr2[i];
        // clip to [0,1]
        res[i] = std::min(res[i], 1.0f);
        res[i] = std::max(res[i], 0.0f);
    }
}

void array_add_noise_gauss(float* arr, float* res, float stdev, int size) {
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> d{0,stdev};

    for (int i = 0; i < size; i++) {
        res[i] = arr[i] + d(gen);
        // clip to [0,1]
        res[i] = std::min(res[i], 1.0f);
        res[i] = std::max(res[i], 0.0f);
    }
}

double array_rms_error(float* arr1, float* arr2, int size) {
    double error = 0;
    for (int i = 0; i < size; i++) {
        error += (arr1[i]-arr2[i]) * (arr1[i]-arr2[i]);
    }
    error /= size;
    error = sqrt(error);
    return error;
}
