#include "Hitable.h"

//__device__ Sphere::Sphere(vec3 cen, double r, Material* m)
//{
//	center = cen;
//	radius = r;
//	mat = m;
//}

//__device__ bool Sphere::hit(const Ray & r, double tmin, double tmax, HitRecord & rec) const
//{
//	vec3 o_c = r.origin() - center;
//	double a = dot(r.direction(), r.direction());
//	double b = 2.0*dot(r.direction(), o_c);
//	double c = dot(o_c, o_c) - radius * radius;
//	double discr = b * b - 4*a*c;
//	if (discr > 0) {
//		float temp = (-b - sqrt(b*b - 4*a * c)) / (2*a);
//		if (temp < tmax && temp > tmin) {
//			//set the hit record variables
//			rec.t = temp;
//			rec.p = r[rec.t];
//			rec.normal = (rec.p - center) / radius;
//			rec.mat_p = mat;
//			return true;
//		}
//		temp = -b + sqrt(b*b - 4*a * c) / (2 * a);
//		if (temp < tmax && temp > tmin) {
//			//set the hit record variables
//			rec.t = temp;
//			rec.p = r[rec.t];
//			rec.normal = (rec.p - center) / radius;
//			rec.mat_p = mat;
//			return true;
//		}
//	}
//	return false;
//}

//__device__ bool hitable_list::hit(const Ray & r, double tmin, double tmax, HitRecord & rec) const
//{
//	HitRecord temp_rec;
//	bool hit_anything = false;
//	double closest_so_far = tmax;
//	for (size_t i = 0; i < list_size; i++)
//	{
//		if ((*p_list)[i]->hit(r,tmin,closest_so_far,temp_rec))
//		{
//			hit_anything = true;
//			closest_so_far = temp_rec.t;
//			rec = temp_rec;
//		}
//	}
//	return hit_anything;
//}
