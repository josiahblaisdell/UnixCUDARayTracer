#include "Material.h"

//__device__ bool Lambertian::scatter(const Ray& r_in, const HitRecord& rec, vec3& attenuation, Ray& scattered, curandState *local_rand_state) const
//{
//	//rec.p + rec.normal gives me the center of the the tangent sphere at p
//	//adding random_in_unit_sphere points in a random direction, hence "scatter"
//	vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
//	scattered = Ray(rec.p, target - rec.p);
//	attenuation = albedo;
//	return true;
//}

//__device__ bool Metal::scatter(const Ray & r_in, const HitRecord & rec, vec3 & attenuation, Ray & scattered, curandState *local_rand_state) const
//{
//	//scatter is in a single direction 
//	vec3 reflected = reflect(unitize(r_in.direction()), rec.normal);
//	scattered = Ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
//	attenuation = albedo;
//	return (dot(scattered.direction() , rec.normal) > 0);
//}

//__device__ bool Dielectric::scatter(const Ray & r_in, const HitRecord & rec, vec3 & attenuation, Ray & scattered, curandState *local_rand_state) const
//{
//	vec3 outward_normal;
//	vec3 reflected = reflect(r_in.direction(), rec.normal);
//	float ni_over_nt;
//	//a glass surface absorbs nothing so attenuation is 1.
//	attenuation = vec3(1., 1., 1.);
//	vec3 refracted;
//	float reflect_prob;
//	float cosine;
//	//if r_in is not parallel to normal
//	if (dot(r_in.direction(), rec.normal) > 0.)
//	{
//		outward_normal = -rec.normal;
//		ni_over_nt = ref_idx;
//		cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
//		cosine = sqrt(1.0f - ref_idx * ref_idx*(1 - cosine * cosine));
//	}
//	else
//	{
//		outward_normal = rec.normal;
//		ni_over_nt = 1 / ref_idx;
//		cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
//	}
//	//if refraction exists
//	if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) {
//		//scattered = Ray(rec.p, refracted);
//		reflect_prob = schlick(cosine, ref_idx);
//	}
//	else {
//		//scattered = Ray(rec.p, reflected);
//		reflect_prob = 1.0;
//	}
//	if (curand_uniform(local_rand_state) < reflect_prob) {
//		scattered = Ray(rec.p, reflected);
//	}
//	else
//	{
//		scattered = Ray(rec.p, refracted);
//	}
//	return true;
//}
