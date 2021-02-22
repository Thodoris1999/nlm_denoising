
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

float* gaussian_kernel(int w_length, float patch_sigma){
    float sum = 0.0;
    float r;
    float s = 2.0*pow(patch_sigma,2);

    float *kernel = (float*)malloc(w_length*w_length*sizeof(float));
    if(!kernel){
        cout << "Couldn't allocate memory for kernel in gaussian_kernel\n";
    }

    int offset = (w_length-1)/2;

    for(int i=-offset;i<=offset;++i){
        for(int j=-offset;j<=offset;++j){
            r = pow(i,2)+pow(j,2);
            kernel[(i+offset)*w_length+(j+offset)] = (exp(-r/s)) / (M_PI*s);
            sum += kernel[(i+offset)*w_length+(j+offset)];
        }
    }

    float kernel_max = kernel[(w_length*w_length-1)/2];

    for(int i=0;i<w_length*w_length;++i){
        kernel[i] /= kernel_max;
    }

    return kernel;
}

float *padded_image(float *img,int w_length,int img_length){

    int extended_length = img_length+w_length-1;
    int half = (w_length-1)/2;

    float *pad_img = (float*)calloc(extended_length*extended_length,sizeof(float));
    if(!pad_img){
        cout << "Couldn't allocate memory for pad_img in padded_image\n";
    }

    //top left corner
    for(int i=0;i<half;++i){
        for(int j=0;j<half;++j){
            pad_img[i*extended_length+j] = img[(half-1-i)*img_length+half-1-j];
        }
    }

    //upper side
    for(int i=0;i<half;++i){
        for(int j=0;j<img_length;++j){
            pad_img[i*extended_length+half+j] = img[(half-1-i)*img_length+j];
        }
    }

    //top right corner
    for(int i=0;i<half;++i){
        for(int j=0;j<half;++j){
            pad_img[i*extended_length+half+img_length+j] = img[(half-i)*img_length-1-j];
        }
    }

    //left side
    for(int i=0;i<img_length;++i){
        for(int j=0;j<half;++j){
            pad_img[(half+i)*extended_length+j] = img[i*img_length+half-1-j];
        }
    }

    //right side
    for(int i=0;i<img_length;++i){
        for(int j=0;j<half;++j){
            pad_img[(half+i)*extended_length+half+img_length+j] = img[(i+1)*img_length-1-j];
        }
    }

    //bottom left corner
    for(int i=0;i<half;++i){
        for(int j=0;j<half;++j){
            pad_img[(half+img_length+i)*extended_length+j] = img[(img_length-1-i)*img_length+half-1-j];
        }
    }

    //bottom side
    for(int i=0;i<half;++i){
        for(int j=0;j<img_length;++j){
            pad_img[(half+img_length+i)*extended_length+half+j] = img[(img_length-1-i)*img_length+j];
        }
    }

    //bottom right corner
    for(int i=0;i<half;++i){
        for(int j=0;j<half;++j){
            pad_img[(half+img_length+i)*extended_length+half+img_length+j] = img[(img_length-i)*img_length-1-j];
        }
    }

    //inside
    for(int i=0;i<img_length;++i){
        for(int j=0;j<img_length;++j){
            pad_img[(half+i)*extended_length+half+j] = img[i*img_length+j];
        }
    }

    return pad_img;
}
