#pragma once
#include <curand_kernel.h>

class RayObject
{
	public:
	__device__ static vec3 random_in_unit_sphere(curandState *local_rand_state) {
		vec3 p(0.0f, 0.0f, 0.0f);

		while (p.squared_length() < 0.0001f) {
			float x = curand_normal(local_rand_state);
			float y = curand_normal(local_rand_state);
			float z = curand_normal(local_rand_state);
			p = vec3(x, y, z);
		}
		p = p / p.squared_length();
		return p;
	}

	__device__ static vec3 random_in_unit_disk(curandState *local_rand_state) {
		vec3 p(0.0f, 0.0f, 0.0f);

		while (p.squared_length() < 0.0001f) {
			float x = curand_normal(local_rand_state);
			float y = curand_normal(local_rand_state);
			float z = 0.0f;
			p = vec3(x, y, z);
		}
		p = p / p.squared_length();
		return p;
	}

	__device__ static vec3 reflect(const vec3& v, const vec3& n) {
		return v - 2 * dot(v, n)*n;
	}

	__device__ static bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
		vec3 uv = unitize(v);
		float dt = dot(uv, n);
		float disc = 1.0 - ni_over_nt * ni_over_nt*(1 - dt * dt);
		//if disc <= 0 then the ray is in the material with the higher refractive index, 
		//		no refraction is possible, all light is reflected
		if (disc > 0) {
			refracted = ni_over_nt * (uv - n * dt) - n * sqrt(disc);
			return true;
		}
		else {
			return false;
		}
	}

	//polynomial expension to compute reflectivity w/r/t angle 
	__device__ static float schlick(float cosine, float ref_idx) {
		float r0 = (1 - ref_idx) / (1 + ref_idx);
		r0 = r0 * r0;
		return r0 + (1 - r0)*pow(1 - cosine, 5);
	}
	
	//(a1,a2,a3) is the line origin
	//(b1,b2,b3) is the line direction
	//t is the point to evaluate at
	//r is the sphube radius
	//s is the sphube squircularity
	__device__ float newtonsmethod(const float t, const float a1, const float a2, const float a3, const float b1, const float b2, const float b3, const float s, const float r) const{
		float div = (1/(r*r));
		float div2 = div*div;
		float numerator = (a1 + b1*t)*(a1 + b1*t) + (a2 + b2*t)*(a2 + b2*t) + (a3 + b3*t)*(a3 + b3*t) - (r*r) 
							- (s*s*(a1 + b1*t)*(a1 + b1*t)*(a2 + b2*t)*(a2 + b2*t))*div 
							- (s*s*(a1 + b1*t)*(a1 + b1*t)*(a3 + b3*t)*(a3 + b3*t))*div 
							- (s*s*(a2 + b2*t)*(a2 + b2*t)*(a3 + b3*t)*(a3 + b3*t))*div 
							+ ((s*s*s*s)*(a1 + b1*t)*(a1 + b1*t)*(a2 + b2*t)*(a2 + b2*t)*(a3 + b3*t)*(a3 + b3*t))*div2;
		float denomenator = 2.0f*b1*(a1 + b1*t) + 2.0f*b2*(a2 + b2*t) + 2.0f*b3*(a3 + b3*t) 
								- (2.0f*b1*(s*s)*(a1 + b1*t)*((a2 + b2*t)*(a2 + b2*t)))*div 
								- (2.0f*b2*(s*s)*((a1 + b1*t)*(a1 + b1*t))*(a2 + b2*t))*div 
								- (2.0f*b1*(s*s)*(a1 + b1*t)*((a3 + b3*t)*(a3 + b3*t)))*div 
								- (2.0f*b3*(s*s)*((a1 + b1*t)*(a1 + b1*t))*(a3 + b3*t))*div 
								- (2.0f*b2*(s*s)*(a2 + b2*t)*((a3 + b3*t)*(a3 + b3*t)))*div 
								- (2.0f*b3*(s*s)*((a2 + b2*t)*(a2 + b2*t))*(a3 + b3*t))*div 
								+ (2.0f*b1*(s*s*s*s)*(a1 + b1*t)*((a2 + b2*t)*(a2 + b2*t))*((a3 + b3*t)*(a3 + b3*t)))*div2 
								+ (2.0f*b2*(s*s*s*s)*((a1 + b1*t)*(a1 + b1*t))*(a2 + b2*t)*((a3 + b3*t)*(a3 + b3*t)))*div2 
								+ (2.0f*b3*(s*s*s*s)*((a1 + b1*t)*(a1 + b1*t))*((a2 + b2*t)*(a2 + b2*t))*(a3 + b3*t))*div2;

		return (float)(t - (numerator/denomenator));
	}
	
	__device__ bool samesign(const float x1, const float x2) const{
		if (fabs(x1) < 0.0001f || fabs(x2) < 0.0001f)
			return true;

		return (x1 >= 0.0f) == (x2 >= 0.0f);
	
	}
};