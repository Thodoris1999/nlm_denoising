#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cassert>

#include "utils.hpp"

using namespace std;
using namespace cv;

//Gaussian kernel of the window
float *gaussian_kernel(int w_length, float patch_sigma){

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

//create the padded image
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

//one command line argument: path to the image
int main(int argc, char* argv[]){

    assert(argc == 2);

    int img_length;
    int img_height;

    int w_length = 5;
    float patch_sigma = 5/3;
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
    array_add_noise_gauss(init_img, noisy_img, 0.05, img_length*img_length);
    show_image(noisy_img, img_length, img_length);

    //start timer
    struct timespec init;
    clock_gettime(CLOCK_MONOTONIC, &init);

    //apply non local means to remove noise
    float* denoised_img = non_local_means(noisy_img,img_length,w_length,filt_sigma,patch_sigma);

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