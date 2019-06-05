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
	
	//p is the polynomial. t is the guess.
	__device__ float newtonsmethod(const float *p,const float *pp, const float t) const{
		float numerator = p[0] + p[1]*t + p[2]*t*t + p[3]*t*t*t + p[4]*t*t*t*t + p[5]*t*t*t*t*t + p[6]*t*t*t*t*t*t;
		float denomenator = pp[0] + pp[1]*t + pp[2]*t*t + pp[3]*t*t*t + pp[4]*t*t*t*t + pp[5]*t*t*t*t*t;
		return (float)(t - (numerator/denomenator));
	}
	
	__device__ bool samesign(const float x1, const float x2) const{
	if (fabs(x1) == 0.0f || fabs(x2) == 0.0f)
        return true;

    return (x1 >= 0.0f) == (x2 >= 0.0f);
	
	}
};