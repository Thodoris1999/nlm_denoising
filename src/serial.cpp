#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include "utils.hpp"
#include "serial.hpp"

using namespace std;
using namespace cv;


float** find_neighborhoods(float* pad_img,int img_length,int w_length,float* g_kernel){

    int pad_length = img_length + w_length - 1;
    int half = (w_length-1)/2;

    float** neighborhoods = (float**)malloc(img_length*img_length*sizeof(float*));
    if(!neighborhoods){
        cout << "Couldn't allocate memory for neighborhoods in find_neighborhoods\n";
    }

    for(int i=0;i<img_length*img_length;++i){
        neighborhoods[i] = (float*)malloc(w_length*w_length*sizeof(float));
        if(!neighborhoods[i]){
            cout << "Couldn't allocate memory for neighborhoods[" <<  i << "] in find_neighborhoods\n";
        }
    }

    //each column containing the neighborhood of the corresponding point
    for(int i=0;i<img_length;++i){
        for(int j=0;j<img_length;++j){
            for(int k=0;k<w_length;++k){
                for(int l=0;l<w_length;++l){
                    neighborhoods[i*img_length+j][k*w_length+l] = pad_img[(i+k)*pad_length+j+l];
                }
            }
        }
    }

    //apply the gaussian patch on each neighborhood
    for(int i=0;i<img_length*img_length;++i){
        for(int j=0;j<w_length*w_length;++j){
            neighborhoods[i][j] *= g_kernel[j];
        }
    }

    return neighborhoods;
}

float* find_xweights(float** neighborhoods,int img_length,int w_length,int x,float filt_sigma){
    
    float* weights = (float*)malloc(img_length*img_length*sizeof(float));
    if(!weights){
        cout << "Couldn't allocate memory for weights in find_xweights\n";
    }

    float z = 0.0;
    float dist;

    for(int i=0;i<img_length*img_length;++i){
        dist = 0.0;
        for(int j=0;j<w_length*w_length;++j){
            dist += pow(neighborhoods[x][j]-neighborhoods[i][j],2); 
        }
        weights[i] = exp(-dist/filt_sigma);
        z += weights[i];
    }

    //divide with the normalizing constant
    for(int i=0;i<img_length*img_length;++i){
        weights[i] /= z;
    }

    return weights;
}


//function applying the non local means filter to an image with noise
float *non_local_means(float *img,int img_length,int w_length, float filt_sigma, float patch_sigma){

    //denoised image to be returned
    float *denoised_img = (float*)calloc(img_length*img_length,sizeof(float));
    if(!denoised_img){
        cout << "Couldn't allocate memory for denoised_img in non_local_means\n";
    }

    //gaussian kernel
    float* g_kernel = gaussian_kernel(w_length,patch_sigma);

    //create image with padding
    float* pad_img = padded_image(img,w_length,img_length);

    //find neighborhood of each point
    float** neighborhoods = find_neighborhoods(pad_img,img_length,w_length,g_kernel);

    //compute the denoised image
    float* x_weights = NULL;

    for(int i=0;i<img_length*img_length;++i){
        x_weights = find_xweights(neighborhoods,img_length,w_length,i,filt_sigma);
        for(int j=0;j<img_length*img_length;++j){
            denoised_img[i] += x_weights[j]*img[j];
        }
        free(x_weights);
        x_weights = NULL;
    }

    //free allocated memory
    free(g_kernel);
    free(pad_img);
    for(int i=0;i<img_length*img_length;++i){
        free(neighborhoods[i]);
    }
    free(neighborhoods);

    return denoised_img;
}

