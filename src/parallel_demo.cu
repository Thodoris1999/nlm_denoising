
#include <iostream>
#include <cstdlib>
#include <cassert>

#include "utils.hpp"
#include "parallel.hpp"

using namespace std;

int main(int argc, char** argv) {
    assert(argc == 2);

    int img_length;
    int img_height;

    int w_length = 5;
    float patch_sigma = 5/3.0;
    float img_noise_stdev = 0.05;
    float filt_sigma = 0.02;

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

    double init_img_error = array_rms_error(noisy_img, init_img, img_length);
    printf("Noisy image RMS error: %lf\n", init_img_error);

    //start timer
    struct timespec init;
    clock_gettime(CLOCK_MONOTONIC, &init);

    // do denoising
    float* denoised_img = cuda_non_local_means(noisy_img,img_length,w_length,filt_sigma,patch_sigma);

    //end timer
    struct timespec last;
    clock_gettime(CLOCK_MONOTONIC, &last);

    struct timespec dur = get_duration(init, last);
    double dur_double = timespec2double(dur);

    double denoised_img_error = array_rms_error(noisy_img, denoised_img, img_length);
    printf("Denoised image RMS error: %lf\n", denoised_img_error);

    printf("Image size: %dx%d, patch size: %dx%d\n",img_length,img_length,w_length,w_length);
    printf("Seconds elapsed: %lf\n", dur_double);

    /*float* denoised_img_serial = non_local_means(noisy_img,img_length,w_length,filt_sigma,patch_sigma);
    double denoised_img_error_serial = array_rms_error(noisy_img, denoised_img_serial, img_length);
    int diff = array_compare(denoised_img_serial, denoised_img, img_length);
    printf("Denoised image (serial method) RMS error: %lf\n", denoised_img_error_serial);
    printf("Denoised image serial-parallel different elements: %d\n", diff);*/

    show_image(denoised_img,img_length,img_length);

    // compare with serial implementation

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
