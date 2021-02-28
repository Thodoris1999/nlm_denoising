# Setup and requirements
This repository depends on CUDA toolkit installation on a machine with a (non ancient) NVidia GPU. You might need to adjust the `nvcc` flags in the `Makefile`
 by changing the target compute version, or remove `-gencode` arguments completely. For example, to test on GTX 940MX, I need to add
 `-gencode arch=compute_50,code=sm_50` (compute version 5.0).
 You can find the compute version of your GPU here https://developer.nvidia.com/cuda-gpus
 
 The `main` branch of this repository also uses OpenCV to read, display and write images. You can download OpenCV on Ubuntu with `apt install libopencv-dev`.
 You may also probably need to change the `Makefile` to point to the path of the headers of your OpenCV installation. That is typically `/usr/include/opencv4`
 if you downloaded through the system package manager and `/usr/local/include/opencv4` if you built from source.
 
 Alterantively, you can use the `cluster` branch, in which the executable handle simple text files with the image's values. Use `img2arr.py` and `arr2img` to 
 convert between images and the text format used by the binaries.
# Builiding
```make all```
# Running
Serial version

```./bin/serial_demo <image_file> <kernel_size> (--show)```

parallel version

```./bin/parallel_demo <image_file> <kernel_size> (--show)```

parallel with shared memory version

```./bin/shared_demo <image_file> <kernel_size> (--show)```


The `--show` argument is optional. Adding it displays the original image, the image with noise added, the noise after denoising is applied,
and the extracted noise obtained as the difference between the original and denoised image (only with OpenCV support).
