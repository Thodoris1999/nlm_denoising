#include <iostream>
#include <cstdlib>
#include <cassert>

#include "utils.hpp"

#define BLOCK_SIZE 16

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

using namespace std;
using namespace cv;

// denoised_img is the output and should be zero-initialized by the user
__global__ void nlm_kernel(float* denoised_img, float* pad_img, int img_size, float* g_kernel, int w_length, float filt_sigma) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = x*img_size + y;
    int pad_length = img_size + w_length - 1;
    int pad_off = w_length/2;

    if (x>=img_size || y>=img_size) return;

    float dist;
    float z = 0;
    denoised_img[idx] = 0;
    for (int i = 0; i < img_size; i++) {
        for (int j = 0; j < img_size; j++) {
            dist = 0;
            for (int k = 0; k < w_length; k++) {
                for (int l = 0; l < w_length; l++) {
                    float val1 = pad_img[(x+k)*pad_length+y+l];
                    float val2 = pad_img[(i+k)*pad_length+j+l];
                    int k_idx = k*w_length + l;
                    dist += g_kernel[k_idx]*g_kernel[k_idx]*(val1-val2)*(val1-val2);
                }
            }
            float weight = expf(-dist/filt_sigma);
            z += weight;

            // apply weighted sum to pixel
            denoised_img[idx] += weight*pad_img[(i+pad_off)*pad_length+j+pad_off];
        }
    }
    // normalize
    denoised_img[idx] /= z;
}

float* cuda_non_local_means(float* img,int img_length,int w_length, float filt_sigma, float patch_sigma){
    float *denoised_img = (float*)malloc(img_length*img_length*sizeof(float));
    if(!denoised_img){
        cout << "Couldn't allocate memory for denoised_img in non_local_means\n";
    }
    float* g_kernel = gaussian_kernel(w_length,patch_sigma);
    float* pad_img = padded_image(img,w_length,img_length);

    float* dg_kernel;
    float* dpad_img;
    float* ddenoised_img;
    gpuErrchk( cudaMalloc(&dg_kernel, w_length*w_length*sizeof(float)) );
    int pad_length = img_length + w_length - 1;
    gpuErrchk( cudaMalloc(&dpad_img, pad_length*pad_length*sizeof(float)) );
    gpuErrchk( cudaMalloc(&ddenoised_img, img_length*img_length*sizeof(float)) );

    // move to device
    gpuErrchk( cudaMemcpy(dg_kernel, g_kernel, w_length*w_length*sizeof(float), cudaMemcpyHostToDevice) );
    gpuErrchk( cudaMemcpy(dpad_img, pad_img, pad_length*pad_length*sizeof(float), cudaMemcpyHostToDevice) );

    // compute weights and apply NLM
    dim3 blockdim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 griddim((img_length+1)/BLOCK_SIZE, (img_length+1)/BLOCK_SIZE);
    nlm_kernel<<<griddim, blockdim>>>(ddenoised_img, dpad_img, img_length, dg_kernel, w_length, filt_sigma);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // move to host
    gpuErrchk( cudaMemcpy(denoised_img, ddenoised_img, img_length*img_length*sizeof(float), cudaMemcpyDeviceToHost) );
    /*for (int i = 0; i < img_length; i++) {
        for (int j = 0; j < img_length; j++) {
            std::cout << denoised_img[i+j*img_length] << " ";
        }
        std::cout << std::endl;
    }*/

    //free allocated memory
    cudaFree(dg_kernel);
    cudaFree(dpad_img);
    cudaFree(ddenoised_img);

    free(g_kernel);
    free(pad_img);

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
