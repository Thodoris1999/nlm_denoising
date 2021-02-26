
#ifndef SERIAL_HPP
#define SERIAL_HPP

float** find_neighborhoods(float* pad_img,int img_length,int w_length,float* g_kernel);
float* find_xweights(float** neighborhoods,int img_length,int w_length,int x,float filt_sigma);
float* non_local_means(float *img,int img_length,int w_length, float filt_sigma, float patch_sigma);

#endif
