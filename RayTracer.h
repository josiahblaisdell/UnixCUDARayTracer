#pragma once
#include <sstream>
#include <fstream>
#include <iostream>
#include <time.h>
#include <float.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <string>
#include "vec3.h"
#include "Ray.h"
#include "Material.h"
#include "Camera.h"
#include "Hitable.h"

#define checkCudaErrors(val) check_cuda((val),#val,__FILE__,__LINE__)
class RayTracer
{
public:
	__host__ RayTracer(int x, int y, int s, int x_block_size, int y_block_size, float squirc );
	__host__ void write_image(std::string fp);
	__host__ void render_kernel(dim3 n, dim3 m);
	__host__ void render_init_kernel(dim3 n, dim3 m);
	__host__ void render_image();
	__host__ ~RayTracer();
	int nx; 
	int ny; 
	int ns; 
	int tx;
	int ty;
	int num_pixels;
	int num_hitables;
	unsigned int frame_buffer_size;
	
	float elapsedTime;
	vec3* frame_buffer;
	//d_rand_state is for rendering the image
	curandState *d_rand_state;
	//d_rand_state2 is for world creation
	curandState *d_rand_state2;
	hitable **d_list;
	hitable **d_world;
	//d_camera defines how the world is viewed
	Camera **d_camera;
	//hitable_list TestScene();
	
};

