
#include "utils.hpp"

#include <cassert>

int main(int argc, char** argv) {
    assert(argc == 2);
    int w, h;
    float* img = read_image(argv[1], &h, &w);
    show_image(img, h, w);

    float* noisy = (float*) malloc(w*h*sizeof(float));
    array_add_noise_gauss(img, noisy, 0.05, w*h);
    show_image(noisy, h, w);

    float* diff = (float*) malloc(w*h*sizeof(float));
    array_subtract(noisy, img, diff, w*h);
    show_image(diff, h, w);

    free(diff);
    free(noisy);
    free(img);
}
