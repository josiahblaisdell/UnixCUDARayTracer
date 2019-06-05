#pragma once
#include "RayObject.h"
#include "Ray.h"



class Camera : public RayObject
{
public:
	__device__ Camera() {}
	//__device__ Camera(float vfov, float aspect);
	//__device__ Camera(vec3 pos, vec3 lookat, vec3 vup, float vfov, float aspect);
	__host__ __device__ Camera(vec3 pos, vec3 lookat, vec3 vup, float vfov, float aspect, float aperture, float focus_dist)
	{
		lens_radius = aperture / 2.0f;

		float theta = vfov * ((float)M_PI) / 180.0f;
		float half_height = (float)tan(theta / 2.0f);
		float half_width = aspect * half_height;
		origin = pos;
		w = unitize(pos - lookat);
		u = unitize(cross(vup, w));
		v = cross(w, u);
		lower_left_corner = origin - half_width*focus_dist*u - half_height*focus_dist*v - focus_dist*w;
		horizontal = 2.0f * half_width*focus_dist*u;
		vertical = 2.0f * half_height*focus_dist*v;
	}

	__device__ Ray GetRay(float s, float t, curandState *local_rand_state){ 
		vec3 rd = lens_radius * random_in_unit_disk(local_rand_state);
		vec3 offset = u * rd.x() + v * rd.y();
		return Ray(origin + offset, lower_left_corner + s * horizontal + t * vertical - origin - offset);
	}

	vec3 origin;
	vec3 lower_left_corner;
	vec3 horizontal;
	vec3 vertical;
	vec3 u, v, w;
	float lens_radius;
};

