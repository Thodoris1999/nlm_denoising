#include <iostream>
#include <cstdlib>
#include <cassert>

#include "utils.hpp"
#include "shared.hpp"

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
//using namespace cv;

// denoised_img is the output and should be zero-initialized by the user
__global__ void nlm_kernel(float* denoised_img, float* pad_img, int img_size, float* g_kernel, int w_length, float filt_sigma) {
    int x = threadIdx.x + blockIdx.x*blockDim.x;
    int y = threadIdx.y + blockIdx.y*blockDim.y;
    int idx = x*img_size + y;
    int pad_length = img_size + w_length - 1;
    int pad_off = w_length/2;

    if (x>=img_size || y>=img_size) return;

    // move gaussian kernel to shared memory
    extern __shared__ float g_kernel_shared[];
    for (int i = 0; i < w_length*w_length / (BLOCK_SIZE*BLOCK_SIZE) + 1; i++) {
        int idx = threadIdx.x + BLOCK_SIZE*threadIdx.y + i * BLOCK_SIZE*BLOCK_SIZE;
        if (idx < w_length*w_length)
            g_kernel_shared[idx] = g_kernel[idx];
    }
    __syncthreads();

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
                    float diff = val1 - val2;
                    int k_idx = k*w_length + l;
                    dist += g_kernel_shared[k_idx]*g_kernel_shared[k_idx]*diff*diff;
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

float* cuda_shared_non_local_means(float* img,int img_length,int w_length, float filt_sigma, float patch_sigma){
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
    nlm_kernel<<<griddim, blockdim, w_length*w_length*sizeof(float)>>>(ddenoised_img, dpad_img, img_length, dg_kernel, w_length, filt_sigma);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk( cudaDeviceSynchronize() );

    // move to host
    gpuErrchk( cudaMemcpy(denoised_img, ddenoised_img, img_length*img_length*sizeof(float), cudaMemcpyDeviceToHost) );

    //free allocated memory
    cudaFree(dg_kernel);
    cudaFree(dpad_img);
    cudaFree(ddenoised_img);

    free(g_kernel);
    free(pad_img);

    return denoised_img;
}

