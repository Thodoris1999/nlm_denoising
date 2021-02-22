#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include "utils.hpp"

using namespace std;
using namespace cv;

float* cuda_non_local_means(float* img,int img_length,int w_length, float filt_sigma, float patch_sigma){
    float *denoised_img = (float*)calloc(img_length*img_length,sizeof(float));
    if(!denoised_img){
        cout << "Couldn't allocate memory for denoised_img in non_local_means\n";
    }

    float* g_kernel = gaussian_kernel(w_length,patch_sigma);
    float* pad_img = padded_image(img,w_length,img_length);

    // find neighborhoods

    // compute weights and apply NLM

    //free allocated memory
    free(g_kernel);
    free(pad_img);
    for(int i=0;i<img_length*img_length;++i){
    //    free(neighborhoods[i]);
    }
    //free(neighborhoods);

    return denoised_img;
}

int main(int argc, char** argv) {
    assert(argc == 2);

    int img_length;
    int img_height;

    int w_length = 5;
    float patch_sigma = 5/3.0;
    float filt_sigma = 0.02;
    float img_noise_stdev = 0.05;

    //initial image
    float* init_img = read_image(argv[1], &img_height, &img_length);   
    if(img_length!=img_height){
        cout << "Not a square image\n";
        exit(-1);
    }   
    show_image(init_img, img_height, img_length);

    //image with noise
    float* noisy_img = (float*) malloc(img_length*img_length*sizeof(float));
    if(!noisy_img){
        cout << "Couldn't allocate memory for noisy_img in main\n";
    }   
    array_add_noise_gauss(init_img, noisy_img, img_noise_stdev, img_length*img_length);
    show_image(noisy_img, img_length, img_length);

    //start timer
    struct timespec init;
    clock_gettime(CLOCK_MONOTONIC, &init);

    // do denoising
    float* denoised_img = cuda_non_local_means(noisy_img,img_length,w_length,filt_sigma,patch_sigma);

    //end timer
    struct timespec last;
    clock_gettime(CLOCK_MONOTONIC, &last);

    long ns;
    uint32_t seconds;
    if(last.tv_nsec <init.tv_nsec){
        ns=init.tv_nsec - last.tv_nsec;
        seconds= last.tv_sec - init.tv_sec -1;
    }

    if(last.tv_nsec >init.tv_nsec){
        ns= last.tv_nsec -init.tv_nsec ;
        seconds= last.tv_sec - init.tv_sec ;
    }
    printf("Image size: %dx%d, patch size: %dx%d\n",img_length,img_length,w_length,w_length);
    printf("Seconds elapsed are %u and the nanoseconds are %ld\n",seconds, ns);

    show_image(denoised_img,img_length,img_length);

    //remainder
    float* diff_img = (float*) malloc(img_length*img_length*sizeof(float));
    if(!diff_img){
        cout << "Couldn't allocate memory for diff_img in main\n";
    }
    array_subtract(noisy_img, denoised_img, diff_img, img_length*img_length);
    show_image(diff_img, img_length, img_length);

    free(init_img);
    free(noisy_img);
    free(denoised_img);
    free(diff_img);
}
