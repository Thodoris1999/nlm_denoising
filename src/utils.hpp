
#ifndef UTILS_HPP
#define UTILS_HPP

#include <string>

#include <opencv2/opencv.hpp>

/**
 * Reads a file from path and returns a float array. The caller is responsible for freeing the
 * image after its use
 *
 * @param   path [input]: Path to the image file
 * @param   h [output]: pointer to image height
 * @param   w [output]: pointer to image width
 */
float* read_image(std::string path, int* h, int* w); 

/**
 * Shows greyscale float image. Blocks execution until user presses any button
 */
void show_image(float* data, int h, int w);

// performs operation on arr1 and arr2 and stores the result on res (user allocated)
void array_add(float* arr1, float* arr2, float* res, int size);
void array_subtract(float* arr1, float* arr2, float* res, int size);
void array_add_noise_gauss(float* arr, float* res, float stdev, int size);

// computer RMS error between the two arrays
double array_rms_error(float* arr1, float* arr2, int size);

// Returns a gaussian kernel of size @param w_length and standard deviant @param patch_sigma
float *gaussian_kernel(int w_length, float patch_sigma);

// Pads an image with mirror edge handling https://en.wikipedia.org/wiki/Kernel_(image_processing)#Edge_Handling
float *padded_image(float *img,int w_length,int img_length);

#endif
