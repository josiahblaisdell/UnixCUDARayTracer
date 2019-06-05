#include <sstream>
#include <fstream>
#include <iostream>
#include <time.h>
#include "RayTracer.h"
#define X_BLOCK_SIZE 8
#define Y_BLOCK_SIZE 8

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line);
__global__ void render(vec3* frame_buffer, int max_x, int max_y);

int main(){
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    int driver_version = 0, runtime_version = 0;
    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    cudaGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf("  Device name: %s\n", prop.name);
    printf("  Memory Clock Rate (KHz): %d\n",
				prop.memoryClockRate);
    printf("  Memory Bus Width (bits): %d\n",
        	    prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n",
				2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf("  Driver Version: %d\n" 
	       "  Runtime Version: %d\n",
	       driver_version, runtime_version);
    }	
    RayTracer RenderedImage(600,400,100,X_BLOCK_SIZE,Y_BLOCK_SIZE);
    RenderedImage.render_image();
    RenderedImage.write_image();
    return 0;
}
