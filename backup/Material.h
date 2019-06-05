#pragma once
#include "vec3.h"
#include "Ray.h"
#include "Hitable.h"
#include <curand_kernel.h>
#include "RayObject.h"
struct HitRecord;

//material will tell us how rays interact with the surface
class Material : public RayObject
{
public:
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState *local_rand_state, bool rand) const = 0;
};

class Lambertian : public Material
{
public:
	__device__ Lambertian(const vec3& a) { albedo = a; }
	//scatter the ray in a random direction
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState *local_rand_state, bool rand) const
	{
		//rec.p + rec.normal gives me the center of the the tangent sphere at p
		//adding random_in_unit_sphere points in a random direction, hence "scatter"
		vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
		scattered = Ray(rec.p, target - rec.p);
		attenuation = albedo;
		return true;
	}
	vec3 albedo;
};

class Metal : public Material
{
public:
	__device__ Metal(const vec3& a, float f) { albedo = a; f < 1.0f ? fuzz = f : fuzz = 1.0f; }
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState *local_rand_state, bool rand) const
	{
		//scatter is in a single direction 
		vec3 reflected = reflect(unitize(r_in.direction()), rec.normal);
		scattered = Ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
		attenuation = albedo;
		return (dot(scattered.direction() , rec.normal) > 0);
	}
	vec3 albedo;
	//radius of sphere used to randomize the reflected direction (to give brushed metal effect)
	//fuzz = 0 is perfectly polished, fuzz of .3 - 1.0 looks like brushed metal
	float fuzz;
};

class Dielectric : public Material 
{
public:
	__device__ Dielectric(float ri) { ref_idx = ri; }
	__device__ virtual bool scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState *local_rand_state, bool rand) const
	{
		float x = curand_uniform(local_rand_state);
		vec3 outward_normal;
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		float ni_over_nt;
		//a glass surface absorbs nothing so attenuation is 1.
		attenuation = vec3(1., 1., 1.);
		vec3 refracted;
		float reflect_prob = 0.0f;
		float cosine;
		Ray temp_scattered;
		//if r_in is not parallel to normal
		if (dot(r_in.direction(), rec.normal) > 0.0f)
		{
			outward_normal = -rec.normal;
			ni_over_nt = ref_idx;
			cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
			cosine = sqrt(1.0f - ref_idx * ref_idx*(1.0f - cosine * cosine));
		}
		else
		{
			outward_normal = rec.normal;
			ni_over_nt = 1.0f / ref_idx;
			cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
		}
		
		float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
		r0 = r0 * r0;
		
		//if refraction exists
		if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) reflect_prob = r0 + (1.0f - r0)*pow(1.0f - cosine, 5.0f);
		else reflect_prob = 1.0f;
		rand < reflect_prob ? scattered = Ray(rec.p, reflected) : scattered = Ray(rec.p, refracted);
		
		
		return true;
	}
	float ref_idx;
};