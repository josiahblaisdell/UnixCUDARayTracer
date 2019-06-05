#include "RayTracer.h"
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line){
	if(result){
		std::cerr << "CUDA Error = " << static_cast<unsigned int>(result) << " at " << file << ":" << line << " '" << func << "' \n";
		//Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}

}

//This function had to be changed alot to reduce the depth of recursion -- otherwise the stack on the graphics card would blow up
__device__ vec3 color(const Ray& r, hitable** world, int max_bounces, curandState *local_rand_state){
	 Ray current_ray = r;
	 //attenuation causes the color to get closer to black every time there is a bounce
	 vec3 current_attenuation = vec3(1.0,1.0,1.0);
	for(int i = 0; i<50; i++){
		HitRecord rec;
		if ((*world)->hit(current_ray, 0.001f, FLT_MAX, rec)){
			Ray scattered;
			vec3 attenuation;
			float x = curand_uniform(local_rand_state);
			if (rec.mat_p->scatter(current_ray, rec, attenuation, scattered,local_rand_state,x)) {
				current_attenuation = current_attenuation*attenuation;
				current_ray = scattered;
			}
			else{
				return vec3(0.0,0.0,0.0);
			}
		}
		else{
			vec3 dir = unitize(current_ray.direction());
			float t = 0.5f*(dir.y() + 1.0f);
			vec3 c = (1.0f - t)*vec3(1., 1., 1.) + t * vec3(.5, .7, 1.);
			return current_attenuation*c;
		}
	}
	//exceeded max number of bounces
	return vec3(0.0,0.0,0.0);
}

__global__ void rand_init(curandState *rand_state) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        curand_init(1984, 0, 0, rand_state);
    }
}

__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    if((i >= max_x) || (j >= max_y)) return;
    int pixel_index = j*max_x + i;
    //Each thread gets same seed, a different sequence number, no offset
	
	//each pixel is associated with a different radom state.
	curand_init(1984, pixel_index,0, &rand_state[pixel_index]);
}

__global__ void render(vec3* frame_buffer, int max_x, int max_y, int ns, Camera **cam, hitable **world, curandState *rand_state){
	//START rendering the image on the GPU
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	//if I am outside the image, stop.
	if((i >= max_x) || (j >= max_y)) return;
	
	int pixel_index = j*max_x + i;
	//get the random state of this pixel
	curandState local_rand_state = rand_state[pixel_index];
	vec3 c(0,0,0);
	
    for(int s=0; s < ns; s++) {
		float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
		float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);
	
		//when j = ny and i = 0 lower_left + u*horizontal + v*verticle = upper_right
		//First ray is shot at upper right corner going across then down.
		Ray ray = (*cam)->GetRay(u, v, &local_rand_state);
		//vec3 p = ray[2.0f];
		c = c + color(ray, world, 50, &local_rand_state);

    }
	rand_state[pixel_index] = local_rand_state;
	c = c / float(ns);
	c = vec3(sqrt(c[0]), sqrt(c[1]), sqrt(c[2]));
	//vec3 icolor(c[0] * 255.99, c[1] * 255.99, c[2] * 255.99);
    frame_buffer[pixel_index] = c;
	//DONE rendering the image on the gpu
}

#define RND (curand_uniform(&local_rand_state))

__global__ void create_world(hitable **d_list, hitable **d_world, Camera **d_camera, int nx, int ny, curandState *rand_state, int num_hitables,float squirc) {
	if (threadIdx.x == 0 && blockIdx.x == 0) {
		//this kernel function will construct the objects in my scene on the GPU (several Spheres and a camera)
		curandState local_rand_state = *rand_state;
		d_list[0] = new Sphere(vec3(0,-1000.0,-1), 1000,
							   new Lambertian(vec3(0.5, 0.5, 0.5)));
		int i = 1;
		float x = RND;
		for(int a = -6; a < 6; a++) {
			for(int b = -6; b < 6; b++) {
				float choose_mat = RND;
				vec3 center(1.7*a+RND,0.2,1.7*b+RND);
				vec3 min(center.x() - .2,center.y() - .2,center.z() - .2);
				vec3 max(center.x() + .2,center.y() + .2,center.z() + .2);
				if(choose_mat < 0.8f) {
					float r1 = RND;

					//Cube *s = new Cube(min, max, new Lambertian(vec3(RND*RND, RND*RND, RND*RND)));
					Sphube *s = new Sphube(center, 0.2f, squirc, new Lambertian(vec3(RND*RND, RND*RND, RND*RND)));
					//Sphere *s = new Sphere(center,0.2f,new Lambertian(vec3(RND*RND, RND*RND, RND*RND)));
					d_list[i++] = s;
				}
				else if(choose_mat < 0.95f) {
					
					//Cube *s = new Cube(min, max, new Metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)),0.5f*RND));
					Sphube *s = new Sphube(center, 0.2f, squirc, new Metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)),0.5f*RND));
					//Sphere *s = new Sphere(center,0.2f,new Metal(vec3(0.5f*(1.0f+RND), 0.5f*(1.0f+RND), 0.5f*(1.0f+RND)),0.5f*RND));
					d_list[i++] = s;
				}
				else {
					//Cube *s = new Cube(min, max, new Dielectric(1.5));
					Sphube *s = new Sphube(center, 0.2f, squirc, new Dielectric(1.5));
					//Sphere *s = new Sphere(center,0.2f,new Dielectric(1.5));
					d_list[i++] = s;
					
				}
			}
		}
		//d_list[i++] = new Cube(vec3(0, -2,-1)  , vec3(1, 2,1), new Dielectric(1.5));
		//d_list[i++] = new Sphere(vec3(-4,1,0), 1.0f,new Lambertian(vec3(0.1, 0.2, 0.5)));
		//d_list[i++] = new Sphere(vec3(0,1,2), 1.0f, new Metal(vec3(0.7, 0.6, 0.5), 0.0));
		//d_list[i++] = new Sphere(vec3(4,1,0), 1.0f, new Dielectric(1.5));
		//d_list[i++] = new Sphube(vec3(0, 1, -1), 1.0f,0.325f, new Metal(vec3(0.7, 0.6, 0.5),0.0f));
		d_list[i++] = new Sphube(vec3(-4,1,0), 1.0f,squirc, new Lambertian(vec3(0.4, 0.2, 0.1)));
		d_list[i++] = new Sphube(vec3(0, 1, 0) , 1.0f, squirc, new Dielectric(1.5));
		d_list[i++] = new Sphube(vec3(4,1,0) , 1.0f, squirc, new Metal(vec3(0.7, 0.6, 0.5),0.0f));
		//d_list[i++] = new Cube(vec3(3.8 -.8, 1-.8, 0-.8),vec3(3.8+.8, 1+.8, 0+.8), new Lambertian(vec3(0.4, 0.2, 0.1)));

		//d_list[i++] = new Cube(vec3(-4-1,1-1,0-1), vec3(-4+1, 1 +1, 0+1),new Metal(vec3(0.7, 0.6, 0.5),0.0f));
		// d_list[0] = new Sphere(vec3(0,0,-1), 0.5,
                                // new Lambertian(vec3(0.1, 0.2, 0.5)));
        // d_list[1] = new Sphere(vec3(0,-100.5,-1), 100,
                                // new Lambertian(vec3(0.8, 0.8, 0.0)));
        // d_list[2] = new Sphere(vec3(1,0,-1), 0.5,
                                // new Metal(vec3(0.8, 0.6, 0.2), 0.0));
        // d_list[3] = new Sphere(vec3(-1,0,-1), 0.5,
                                // new Dielectric(1.5));
        // d_list[4] = new Sphere(vec3(-1,0,-1), -0.45,
                                // new Dielectric(1.5));
		//*rand_state = local_rand_state;
		*d_world  = new hitable_list(d_list, num_hitables);

		vec3 lookfrom(13,2,3);
		vec3 lookat(0,0,0);
		float dist_to_focus = (lookfrom - vec3(0,0,0)).length(); //(lookfrom-lookat).length();
		float aperture = 0.01;
		*d_camera   = new Camera(lookfrom,lookat,vec3(0,1,0),20.0,float(nx)/float(ny),aperture,10.0);
	}
}



__global__ void free_world(hitable **d_list, hitable **d_world, Camera **d_camera, int num_hitables) {
    for(int i=0; i < num_hitables; i++) {
        delete ((Sphube *)d_list[i])->mat;
        delete d_list[i];
    }
    //delete *d_world;
    //delete *d_camera;
}



RayTracer::RayTracer(int x, int y, int s, int x_block_size, int y_block_size, float squirc){
	nx = x;
	ny = y;
	ns = s;
	tx = x_block_size;
	ty = y_block_size;
	num_pixels = nx*ny;
	frame_buffer_size = num_pixels*sizeof(vec3);
	//allocate frame buffer on device
	checkCudaErrors(cudaMallocManaged((void**)&frame_buffer,frame_buffer_size));
	//allocate random state on device
	checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
	checkCudaErrors(cudaMalloc((void **)&d_rand_state2, 1*sizeof(curandState)));
	num_hitables = 12*12 + 3 + 1;//6*6 + 3 + 1; //4;//22*22+1+3; 
	//22*22+1+3;
	
	//initialize random states on device
	rand_init<<<1,1>>>(d_rand_state2);
	checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	//allocate hitables on device
	checkCudaErrors(cudaMalloc((void **)&d_list, num_hitables*sizeof(hitable *)));
	//allocate world on device
	checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hitable *)));
	//allocate camera on device 
	checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(Camera *)));
	
	//initialize the world on the device 
	create_world<<<1,1>>>(d_list, d_world, d_camera, nx, ny, d_rand_state2,num_hitables,squirc);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void RayTracer::write_image(std::string fp){
	std::string filepath = fp;
	std::ofstream of;
	of.open(filepath.c_str(), std::ios_base::app);
	of << "P3\n" << nx << " " << ny << "\n255\n";
	for(int j = ny-1; j>= 0; j--){
		for(int i = 0; i < nx; i++){
			unsigned int pixel_index = j*nx + i;
			vec3 c = frame_buffer[pixel_index];
			int ir = int(255.99f*c.r());
			int ig = int(255.99f*c.g());
			int ib = int(255.99f*c.b());
			of << ir << " " << ig << " " << ib << "\n";
		}
	}
	of.close();
	
	    // clean up
    checkCudaErrors(cudaDeviceSynchronize());

}


__host__ RayTracer::~RayTracer(){
    // free_world<<<1,1>>>(d_list,d_world,d_camera,num_hitables);
	// checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaGetLastError());
	// checkCudaErrors(cudaDeviceSynchronize());
    // //checkCudaErrors(cudaFree(*d_camera));
	// checkCudaErrors(cudaDeviceSynchronize());
    // //checkCudaErrors(cudaFree(*d_world));
	// checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaFree(*d_list));
	// checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaFree(d_rand_state));
	// checkCudaErrors(cudaDeviceSynchronize());
    // checkCudaErrors(cudaFree(frame_buffer));
	checkCudaErrors(cudaDeviceSynchronize());
	cudaDeviceReset();
	checkCudaErrors(cudaDeviceSynchronize());
}

__host__ void RayTracer::render_kernel(dim3 n, dim3 m){
	//cudaEvent_t start1, stop1;
	//float t_elapsed;
	//checkCudaErrors(cudaEventCreate(&start1));
	//checkCudaErrors(cudaEventCreate(&stop1));
	//checkCudaErrors(cudaEventRecord(start1));
	render<<<n, m>>>(frame_buffer, nx, ny,  ns, d_camera, d_world, d_rand_state);
	//checkCudaErrors(cudaEventRecord(stop1));
	//checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
	//checkCudaErrors(cudaEventSynchronize(stop1));
	//checkCudaErrors(cudaEventSynchronize(stop1));
	//checkCudaErrors(cudaEventElapsedTime(&t_elapsed,start1,stop1));
	//checkCudaErrors(cudaEventDestroy(start1));
	//checkCudaErrors(cudaEventDestroy(stop1));
	//elapsedTime += t_elapsed;
}

__host__ void RayTracer::render_init_kernel(dim3 n, dim3 m){
	cudaEvent_t start, stop;
	float t_elapsed2;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start));
	render_init<<<n, m>>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaEventSynchronize(stop));
	checkCudaErrors(cudaEventElapsedTime(&t_elapsed2,start,stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	elapsedTime += t_elapsed2;
}

__host__ void RayTracer::render_image(){
	vec3 lookfrom(13, 2, 3);
	vec3 lookat(0, 0, 0);
	double dist_to_focus = 10.0;
	double aperture = 0.1;
	Camera cam(lookfrom, lookat, vec3(0, 1, 0), 30.0f, float(nx) / float(ny), aperture, dist_to_focus);
	dim3 blocks(nx/tx+1,ny/ty +1);
	dim3 threads(tx,ty);
	//start and stop used to measure performance
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	//RENDER THE SCENE HERE
	checkCudaErrors(cudaEventRecord(start));
	render_init<<<blocks, threads>>>(nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

	render<<<blocks, threads>>>(frame_buffer, nx, ny,  ns, d_camera, d_world, d_rand_state);
	checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventRecord(stop));
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime,start,stop));
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));
	//DONE RENDERING THE SCENE HERE
	elapsedTime /= 10000; //convert ms to s.
	printf("  Time Elapsed: %10.2f\n",elapsedTime);
	printf("  Pixels/Second: %10.2f\n",num_pixels/elapsedTime);
	printf("  Rays/Second: %10.2f\n",num_pixels*ns/elapsedTime);
	//checkCudaErrors(cudaGetLastError());
	//checkCudaErrors(cudaDeviceSynchronize());
}

