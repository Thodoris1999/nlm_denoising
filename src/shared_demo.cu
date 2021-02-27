
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>
#include <sstream>

#include "utils.hpp"

#include "shared.hpp"

using namespace std;
//using namespace cv;

int main(int argc, char* argv[]){
    assert(argc > 2);
    bool visualize = false;
    if (argc > 3 && strcmp(argv[3], "--show") == 0) {
        visualize = true;
    }

    int img_length;
    int img_height;

    int w_length = atoi(argv[2]);
    float patch_sigma = 5/3.0;
    float filt_sigma = 0.02;
    float img_noise_stdev = 0.05;

    //initial image
    float* init_img = read_image(argv[1], &img_height, &img_length);
    if(img_length!=img_height){
        cout << "Not a square image\n";
        exit(-1);
    }
    if (visualize) show_image(init_img, img_height, img_length);

    // time, RMSE, denoised image
    string imagename = string(argv[1]).substr(0, string(argv[1]).size()-4);
    std::stringstream info_ss;
    info_ss << imagename << "_" << w_length << "_shared_info.txt";
    FILE* fp = fopen(info_ss.str().c_str(), "a+");

    //image with noise
    float* noisy_img = (float*) malloc(img_length*img_length*sizeof(float));
    if(!noisy_img){
        cout << "Couldn't allocate memory for noisy_img in main\n";
    }
    array_add_noise_gauss(init_img, noisy_img, img_noise_stdev, img_length*img_length);
    if (visualize) show_image(noisy_img, img_length, img_length);

    for (int i = 0; i < 5; i++) {
        float h = filt_sigma + (i-2)*0.005;
        //start timer
        struct timespec init;
        clock_gettime(CLOCK_MONOTONIC, &init);

        //apply non local means to remove noise
        float* denoised_img = cuda_shared_non_local_means(noisy_img,img_length,w_length,h,patch_sigma);

        //end timer
        struct timespec last;
        clock_gettime(CLOCK_MONOTONIC, &last);

        struct timespec dur = get_duration(init, last);
        double dur_double = timespec2double(dur);

        printf("Image size: %dx%d, patch size: %dx%d\n",img_length,img_length,w_length,w_length);
        if (i == 0) {
            fprintf(fp, "%lf\n", dur_double);
        }

        double denoised_img_error = array_rms_error(noisy_img, denoised_img, img_length);
        printf("Denoised image RMS error: %lf\n", denoised_img_error);
        fprintf(fp, "%f %lf\n", h, denoised_img_error);

        if (visualize) show_image(denoised_img,img_length,img_length);

        //remainder
        float* diff_img = (float*) malloc(img_length*img_length*sizeof(float));
        if(!diff_img){
            cout << "Couldn't allocate memory for diff_img in main\n";
        }
        array_subtract(noisy_img, denoised_img, diff_img, img_length*img_length);
        if (visualize) show_image(diff_img, img_length, img_length);

        std::stringstream dimg_ss, diffimg_ss;
        dimg_ss << imagename << "_" << h << "_" << w_length << "shared_denoised.txt";
        diffimg_ss << imagename <<  "_" << h << "_" << w_length << "shared_diff.txt";
        write_image(dimg_ss.str().c_str(), denoised_img, img_length);
        write_image(diffimg_ss.str().c_str(), diff_img, img_length);
        std::cout << dimg_ss.str().c_str() << std::endl;

        free(denoised_img);
        free(diff_img);
    }

    fclose(fp);
    free(init_img);
    free(noisy_img);
}
